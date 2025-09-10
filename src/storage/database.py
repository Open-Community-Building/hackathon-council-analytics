import couchdb
from datetime import datetime


class CouchDBLogger:
    """
    A class to handle CouchDB connection and logging of document processing status.
    """

    def __init__(self, config: dict, secrets: dict) -> None:
        couch_conf = secrets.get("couchdb", {})
        couch_url = couch_conf.get("url", "http://localhost:5984/")
        couch_user = couch_conf.get("user")
        couch_pass = couch_conf.get("password")
        self.couch = couchdb.Server(couch_url)

        if couch_user and couch_pass:
            self.couch.resource.credentials = (couch_user, couch_pass)

        db_name = couch_conf.get("database", "rag_documents")
        if db_name not in self.couch:
            self.db = self.couch.create(db_name)
        else:
            self.db = self.couch[db_name]
        print(f"Connected to CouchDB database: {self.db}")

    def log_status(self, filename: str, status: str, extra: dict | None = None) -> None:
        """
        Create or update a document with its processing status.
        """
        now = datetime.utcnow().isoformat()
        doc_id = filename  # alternativ uuid.uuid4().hex

        doc = self.db.get(doc_id)
        print(f"Logging status for {filename}: {status} to {doc}")
        if doc:
            doc["status"] = status
            doc["updated_at"] = now
            if extra:
                doc.update(extra)
            self.db.save(doc)
        else:
            new_doc = {
                "_id": doc_id,
                "filename": filename,
                "status": status,
                "created_at": now,
                "updated_at": now,
            }
            if extra:
                new_doc.update(extra)
            self.db.save(new_doc)
