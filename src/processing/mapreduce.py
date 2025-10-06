"""MapReduce implementation for text processing and feature extraction."""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, split, explode, lower, regexp_replace, trim, size
from pyspark.sql.types import StringType, IntegerType, ArrayType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MapReduceProcessor:
    """MapReduce processor for text analysis and feature extraction."""
    
    def __init__(self, spark: SparkSession, config):
        """Initialize MapReduce processor."""
        self.spark = spark
        self.config = config
        self.word_counts = None
        self.vocabulary = None
    
    def map_phase(self, df: DataFrame) -> DataFrame:
        """
        Map phase: Tokenize text and emit (word, 1) pairs.
        
        Args:
            df: Input DataFrame with 'text' column
            
        Returns:
            DataFrame with tokenized words
        """
        logger.info("Starting Map phase: Tokenization")
        
        # Tokenize text
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        df_tokenized = tokenizer.transform(df)
        
        # Remove stopwords if configured
        if self.config.text_processing.remove_stopwords:
            remover = StopWordsRemover(
                inputCol="words", 
                outputCol="filtered_words",
                stopWords=self._get_stopwords()
            )
            df_tokenized = remover.transform(df_tokenized)
            words_col = "filtered_words"
        else:
            words_col = "words"
        
        # Filter words by length
        df_tokenized = df_tokenized.withColumn(
            words_col,
            col(words_col).cast(ArrayType(StringType()))
        )
        
        # Explode words to create (word, 1) pairs
        df_words = df_tokenized.select(
            col("label"),
            explode(col(words_col)).alias("word")
        )
        
        # Clean and filter words
        df_words = df_words.withColumn("word", lower(trim(col("word"))))
        df_words = df_words.withColumn("word", regexp_replace(col("word"), r'[^\w]', ''))
        
        # Filter by word length
        min_len = self.config.text_processing.min_word_length
        max_len = self.config.text_processing.max_word_length
        df_words = df_words.filter(
            (col("word").rlike(f'^.{{{min_len},{max_len}}}$')) &
            (col("word") != "")
        )
        
        # Add count column for reduce phase
        df_words = df_words.withColumn("count", col("word").cast(IntegerType()).cast(IntegerType()))
        df_words = df_words.withColumn("count", col("count") * 0 + 1)  # Set all counts to 1
        
        logger.info(f"Map phase completed. Generated {df_words.count()} word tokens")
        return df_words
    
    def reduce_phase(self, df_words: DataFrame) -> DataFrame:
        """
        Reduce phase: Sum counts for each word.
        
        Args:
            df_words: DataFrame from map phase with (word, count) pairs
            
        Returns:
            DataFrame with (word, total_count) pairs
        """
        logger.info("Starting Reduce phase: Word count aggregation")
        
        # Group by word and sum counts
        word_counts = df_words.groupBy("word").sum("count").alias("total_count")
        word_counts = word_counts.select(
            col("word"),
            col("sum(count)").alias("total_count")
        )
        
        # Filter by document frequency
        min_df = self.config.text_processing.min_document_frequency
        max_df_ratio = self.config.text_processing.max_document_frequency
        
        # Calculate document frequency
        total_docs = df_words.select("label").distinct().count()
        min_doc_count = max(min_df, int(total_docs * 0.01))  # At least 1% of documents
        max_doc_count = int(total_docs * max_df_ratio)
        
        # Filter words by document frequency
        doc_freq = df_words.groupBy("word").agg(
            col("word").alias("word"),
            col("label").distinct().count().alias("doc_freq")
        )
        
        word_counts = word_counts.join(doc_freq, "word")
        word_counts = word_counts.filter(
            (col("doc_freq") >= min_doc_count) & 
            (col("doc_freq") <= max_doc_count)
        )
        
        # Sort by frequency and limit vocabulary size
        max_vocab = self.config.text_processing.max_vocab_size
        word_counts = word_counts.orderBy(col("total_count").desc()).limit(max_vocab)
        
        self.word_counts = word_counts
        logger.info(f"Reduce phase completed. Vocabulary size: {word_counts.count()}")
        
        return word_counts
    
    def build_vocabulary(self, word_counts: DataFrame) -> List[str]:
        """
        Build vocabulary from word counts.
        
        Args:
            word_counts: DataFrame with word counts
            
        Returns:
            List of vocabulary words
        """
        logger.info("Building vocabulary from word counts")
        
        vocabulary = [row["word"] for row in word_counts.select("word").collect()]
        self.vocabulary = vocabulary
        
        logger.info(f"Vocabulary built with {len(vocabulary)} words")
        return vocabulary
    
    def extract_features(self, df: DataFrame, vocabulary: List[str] = None) -> DataFrame:
        """
        Extract features using TF-IDF and bag-of-words.
        
        Args:
            df: Input DataFrame with 'text' column
            vocabulary: Optional vocabulary list
            
        Returns:
            DataFrame with extracted features
        """
        logger.info("Extracting features using TF-IDF and bag-of-words")
        
        # Tokenize text
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        
        # Remove stopwords if configured
        if self.config.text_processing.remove_stopwords:
            remover = StopWordsRemover(
                inputCol="words", 
                outputCol="filtered_words",
                stopWords=self._get_stopwords()
            )
            words_col = "filtered_words"
        else:
            remover = None
            words_col = "words"
        
        # Create pipeline stages
        stages = [tokenizer]
        if remover:
            stages.append(remover)
        
        # Add CountVectorizer for bag-of-words
        if self.config.features.use_bow:
            cv = CountVectorizer(
                inputCol=words_col,
                outputCol="bow_features",
                vocabSize=self.config.features.max_features,
                minDF=self.config.text_processing.min_document_frequency
            )
            stages.append(cv)
        
        # Add IDF for TF-IDF
        if self.config.features.use_tfidf:
            idf = IDF(
                inputCol="bow_features" if self.config.features.use_bow else words_col,
                outputCol="tfidf_features"
            )
            stages.append(idf)
        
        # Create and fit pipeline
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(df)
        
        # Transform data
        df_features = model.transform(df)
        
        logger.info("Feature extraction completed")
        return df_features
    
    def _get_stopwords(self) -> List[str]:
        """Get stopwords list."""
        # Basic English stopwords
        stopwords = [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "i", "you", "we", "they", "this",
            "these", "those", "have", "had", "do", "does", "did", "can",
            "could", "would", "should", "may", "might", "must", "shall"
        ]
        return stopwords
    
    def get_word_statistics(self, word_counts: DataFrame) -> Dict:
        """Get word statistics from word counts."""
        stats = word_counts.agg(
            col("total_count").sum().alias("total_words"),
            col("total_count").avg().alias("avg_word_freq"),
            col("total_count").max().alias("max_word_freq"),
            col("total_count").min().alias("min_word_freq")
        ).collect()[0]
        
        return {
            "total_words": stats["total_words"],
            "avg_word_freq": stats["avg_word_freq"],
            "max_word_freq": stats["max_word_freq"],
            "min_word_freq": stats["min_word_freq"],
            "vocabulary_size": word_counts.count()
        }
