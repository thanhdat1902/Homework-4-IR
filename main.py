from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class InvertedIndexSearchEngine:
    def __init__(self):
        # Initialize MongoDB and collections
        self.db = self._connect_to_mongodb()
        self.terms_collection = self.db['terms']         # Inverted index
        self.documents_collection = self.db['documents']  # Documents collection
        self.document_count = 0
        self.vectorizer = None
        self.vocabulary = {}

        # Clear existing collections
        self.terms_collection.delete_many({})
        self.documents_collection.delete_many({})

    def _connect_to_mongodb(self):
        """Establish connection to MongoDB."""
        db_name = "search_engine"
        try:
            client = MongoClient(host="localhost", port=27017)
            return client[db_name]
        except Exception as e:
            print("Failed to connect to MongoDB:", e)

    def add_new_document(self, content):
        """Add a new document to the database."""
        self.documents_collection.insert_one({"_id": self.document_count, "content": content})
        self.document_count += 1

    def build_inverted_index(self):
        """Generate an inverted index and store it in MongoDB."""
        # Fetch documents
        documents = [doc['content'] for doc in self.documents_collection.find()]
        if not documents:
            print("No documents found to process.")
            return

        # Generate TF-IDF matrix with unigrams, bigrams, and trigrams
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        tfidf_matrix = self.vectorizer.fit_transform(documents)

        # Save document vectors and vocabulary
        self.document_vectors = tfidf_matrix.toarray()
        self.vocabulary = self.vectorizer.vocabulary_

        # Build the inverted index
        unique_id = 0  # Counter for unique _id
        for term, pos in self.vocabulary.items():
            term_data = {
                "_id": unique_id,  # Unique ID for each term
                "term": term,  # Include term text for clarity
                "pos": pos,  # Position in vocabulary dictionary
                "docs": []  # List of documents with TF-IDF values
            }
            for doc_idx in range(tfidf_matrix.shape[0]):
                tfidf_value = tfidf_matrix[doc_idx, pos]
                if tfidf_value > 0:
                    term_data["docs"].append({"doc_id": doc_idx, "tfidf": tfidf_value})
            self.terms_collection.insert_one(term_data)
            unique_id += 1  # Increment for the next term

        print("Inverted index has been successfully built.")

    def rank_documents(self, search_query):
        # transform query using learned vocabulary and document frequencies
        query_vector = self.vectorizer.transform([search_query]).toarray()[0]

        # calculate cosine similarity for each query/document pair
        scores = []
        num_docs = self.documents_collection.count_documents({})
        for doc_id in range(num_docs):
            cosine_score = cosine_similarity([query_vector, self.document_vectors[doc_id]])[0][1]
            scores.append((doc_id, cosine_score))

        # sort documents by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        for doc_id, score in scores:
            if score > 0:
                document = self.documents_collection.find_one({"_id": doc_id})
                print(f"Document: \"{document['content']}\", - Score: {score}")


if __name__ == '__main__':
    # Initialize search engine
    invertedIndexSearchEngine = InvertedIndexSearchEngine()

    # Add sample documents
    invertedIndexSearchEngine.add_new_document("After the medication, headache and nausea were reported by the patient.")
    invertedIndexSearchEngine.add_new_document("The patient reported nausea and dizziness caused by the medication.")
    invertedIndexSearchEngine.add_new_document("Headache and dizziness are common effects of this medication.")
    invertedIndexSearchEngine.add_new_document("The medication caused a headache and nausea, but no dizziness was reported.")

    # Build the inverted index
    invertedIndexSearchEngine.build_inverted_index()

    # Perform queries
    queries = ["nausea and dizziness", "effects", "nausea was reported", "dizziness", "the medication"]
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        print("-------Ranking------")
        invertedIndexSearchEngine.rank_documents(query)
