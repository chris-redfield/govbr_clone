import requests
from typing import Dict, Optional, List
from dataclasses import dataclass
from urllib.parse import urljoin

@dataclass
class Collection:
    id: str
    name: str

@dataclass
class Field:
    id: str
    name: str
    type: str
    operation: Optional[str]

class CogfyClient:
    def __init__(self, api_key: str, base_url: str):
        """Initialize the Cogfy client with an API key.

        Args:
            api_key (str): Your Cogfy API key
            base_url (str): The base URL for the Cogfy API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Api-Key": api_key
        }

    def create_chat(self, collection_id: str, field_id: str):
        """Creates new chat for message sending
        """

        endpoint = f"collections/{collection_id}/chats"
        url = urljoin(f"{self.base_url}/", endpoint)

        params = {
            "fieldId": field_id
        }

        response = requests.post(url, headers=self.headers, json=params)
        response.raise_for_status()  # Raise an exception for bad status codes

        return response.json()

    def send_chat_message(self, collection_id: str, chat_id: str, content: str):
        """Send a message to an already created chat
        """

        # POST https://api.cogfy.com/collections/:collectionId/chats/:chatId/messages
        endpoint = f"collections/{collection_id}/chats/{chat_id}/messages"
        url = urljoin(f"{self.base_url}/", endpoint)

        params = {
            "content": content,
            "contentType": "text",
        }
        response = requests.post(url, headers=self.headers, json=params)
        response.raise_for_status()  # Raise an exception for bad status codes

        return response.json()

    def find_chat(self, collection_id: str, chat_id: str):
        """Sends chat message to the API
        """

        endpoint = f"collections/{collection_id}/chats/{chat_id}"
        url = urljoin(f"{self.base_url}/", endpoint)

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()  # Raise an exception for bad status codes

        return response.json()

    def list_collections(self, page_number: int = 0, page_size: int = 10) -> Dict:
        """List collections from the Cogfy server.

        Args:
            page_number (int): The page number to retrieve (starts at 0)
            page_size (int): Number of results per page

        Returns:
            Dict: Response containing collections data and pagination info
        """
        url = urljoin(f"{self.base_url}/", "collections")
        params = {
            "pageNumber": page_number,
            "pageSize": page_size
        }

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes

        return response.json()

    def find_collection(self, collection_id: str) -> Optional[Collection]:
        """Find a collection by its ID.

        Args:
            collection_id (str): The ID of the collection to find

        Returns:
            Optional[Collection]: The collection if found, None otherwise
        """
        url = urljoin(f"{self.base_url}/", f"collections/{collection_id}")

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return Collection(id=data["id"], name=data["name"])
        except requests.exceptions.RequestException:
            return None

    def get_collection_id(self, collection_name: str) -> Optional[str]:
        """Find a collection ID by its name.

        Args:
            collection_name (str): The name of the collection to find

        Returns:
            Optional[str]: The collection ID if found, None otherwise
        """
        page_number = 0
        page_size = 100  # Use a larger page size to reduce number of requests

        while True:
            result = self.list_collections(page_number=page_number, page_size=page_size)

            # Search in current page
            for collection in result["data"]:
                if collection["name"] == collection_name:
                    return collection["id"]

            # Check if we've reached the last page
            if (page_number + 1) * page_size >= result["totalSize"]:
                break

            page_number += 1

        return None

    def list_fields(self, collection_id: str) -> List[Field]:
        """List all fields in a collection.

        Args:
            collection_id (str): The ID of the collection to list fields from

        Returns:
            List[Field]: List of fields in the collection
        """
        url = urljoin(f"{self.base_url}/", f"collections/{collection_id}/fields")

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        data = response.json()
        return [Field(**field) for field in data["data"]]

    def create_field(self, collection_id: str, name: str, field_type: str) -> Field:
        """Create a new field in a collection.

        Args:
            collection_id (str): The ID of the collection to add the field to
            name (str): The name of the new field
            field_type (str): The type of the new field (e.g., "text", "number", "boolean", "datetime")

        Returns:
            Field: The created field

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        url = urljoin(f"{self.base_url}/", f"collections/{collection_id}/fields")
        payload = {
            "name": name,
            "type": field_type
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        # The POST API only returns the field ID
        field_id = response.json()["id"]

        # Fetch the complete field details
        fields = self.list_fields(collection_id)
        for field in fields:
            if field.id == field_id:
                return field

        raise ValueError(f"Created field {field_id} not found in collection {collection_id}")

    def create_record(self, collection_id: str, properties: Dict[str, Dict]) -> Dict[str, str]:
        """Create a record in a collection.

        Args:
            collection_id (str): The ID of the collection to create the record in
            properties (Dict[str, Dict]): Dictionary mapping field IDs to their values and types
                Example: {
                    "field_id": {
                        "type": "text",
                        "text": {"value": "Text value"}
                    }
                }
                For datetime fields, use:
                {
                    "field_id": {
                        "type": "datetime",
                        "datetime": {"value": "2024-03-20T15:30:00Z"}
                    }
                }

        Returns:
            Dict[str, str]: Dictionary containing the created record's ID
                Example: {"id": "c2fb90f0-3d48-4d71-af78-1c09b39923a4"}

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        url = urljoin(f"{self.base_url}/", f"collections/{collection_id}/records")
        payload = {"properties": properties}

        response = requests.post(url, headers=self.headers, json=payload)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"Error creating record: {e}")
            print(f"Response: {response.text}")
            raise e

        return response.json()

    def query_records(
        self,
        collection_id: str,
        filter: Optional[Dict] = None,
        order_by: Optional[List[Dict]] = None,
        page_number: int = 0,
        page_size: int = 50
    ) -> Dict:
        """Query records in a collection with filtering and ordering.

        Args:
            collection_id (str): The ID of the collection to query
            filter (Optional[Dict]): Filter criteria for the query
                Example: {
                    "type": "and",
                    "and": {
                        "filters": [
                            {
                                "type": "equals",
                                "equals": {
                                    "fieldId": "field_id",
                                    "value": "some_value"
                                }
                            }
                        ]
                    }
                }
            order_by (Optional[List[Dict]]): List of ordering criteria
                Example: [
                    {
                        "fieldId": "field_id",
                        "direction": "asc"
                    }
                ]
            page_number (int): The page number to retrieve (starts at 0)
            page_size (int): Number of results per page

        Returns:
            Dict: Response containing records data and pagination info
                Example: {
                    "data": [
                        {
                            "id": "record_id",
                            "properties": {
                                "field_id": {
                                    "type": "text",
                                    "text": {"value": "value"}
                                }
                            }
                        }
                    ],
                    "pageNumber": 0,
                    "pageSize": 50,
                    "totalSize": 1
                }

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        url = urljoin(f"{self.base_url}/", f"collections/{collection_id}/records/query")

        payload = {
            "pageNumber": page_number,
            "pageSize": page_size
        }

        if filter:
            payload["filter"] = filter

        if order_by:
            payload["orderBy"] = order_by

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()

class CollectionManager:
    def __init__(self, client: CogfyClient, collection_identifier: str):
        """Initialize the CollectionManager with a collection name or ID.

        Args:
            client (CogfyClient): An initialized CogfyClient instance
            collection_identifier (str): Either the collection name or ID
        """
        self.client = client
        self.collection_id = None
        self.collection_name = None

        # Check if the identifier looks like a UUID (collection ID)
        if len(collection_identifier) == 36 and collection_identifier.count('-') == 4:
            # Try to find the collection by ID
            collection = client.find_collection(collection_identifier)
            if collection:
                self.collection_id = collection.id
                self.collection_name = collection.name
            else:
                raise ValueError(f"Collection not found with ID: {collection_identifier}")
        else:
            # Assume it's a collection name
            self.collection_name = collection_identifier
            self.collection_id = client.get_collection_id(collection_identifier)

        if not self.collection_id:
            raise ValueError(f"Collection not found: {collection_identifier}")

    @property
    def id(self) -> str:
        """Get the collection ID.

        Returns:
            str: The collection ID
        """
        return self.collection_id

    @property
    def name(self) -> str:
        """Get the collection name.

        Returns:
            str: The collection name
        """
        return self.collection_name

    def list_columns(self) -> List[Field]:
        """List all fields (columns) in the collection.

        Returns:
            List[Field]: List of fields in the collection
        """
        return self.client.list_fields(self.collection_id)

    def ensure_fields(self, fields: Dict[str, str]) -> List[Field]:
        """Ensure that all specified fields exist in the collection, creating any that don't.

        Args:
            fields (Dict[str, str]): Dictionary mapping field names to their types.
                Supported types: "text", "number", "boolean", "datetime"

        Returns:
            List[Field]: List of all fields in the collection after ensuring the specified ones exist
        """
        # Get current fields
        current_fields = self.list_columns()
        current_field_names = {field.name for field in current_fields}

        # Create any missing fields
        for name, field_type in fields.items():
            if name not in current_field_names:
                self.client.create_field(self.collection_id, name, field_type)

        # Return updated list of fields
        return self.list_columns()

    def create_record(self, properties: Dict[str, Dict]) -> str:
        """Create a record in the managed collection.

        Args:
            properties (Dict[str, Dict]): Dictionary mapping field IDs to their values and types
                Example: {
                    "field_id": {
                        "type": "text",
                        "text": {"value": "Text value"}
                    }
                }
                For datetime fields, use:
                {
                    "field_id": {
                        "type": "datetime",
                        "datetime": {"value": "2024-03-20T15:30:00Z"}
                    }
                }

        Returns:
            str: The ID of the created record

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        response = self.client.create_record(self.collection_id, properties)
        return response["id"]

    def query_records(
        self,
        filter: Optional[Dict] = None,
        order_by: Optional[List[Dict]] = None,
        page_number: int = 0,
        page_size: int = 50
    ) -> Dict:
        """Query records in the managed collection with filtering and ordering.

        Args:
            filter (Optional[Dict]): Filter criteria for the query
                Example: {
                    "type": "and",
                    "and": {
                        "filters": [
                            {
                                "type": "equals",
                                "equals": {
                                    "fieldId": "field_id",
                                    "value": "some_value"
                                }
                            }
                        ]
                    }
                }
            order_by (Optional[List[Dict]]): List of ordering criteria
                Example: [
                    {
                        "fieldId": "field_id",
                        "direction": "asc"
                    }
                ]
            page_number (int): The page number to retrieve (starts at 0)
            page_size (int): Number of results per page

        Returns:
            Dict: Response containing records data and pagination info
                Example: {
                    "data": [
                        {
                            "id": "record_id",
                            "properties": {
                                "field_id": {
                                    "type": "text",
                                    "text": {"value": "value"}
                                }
                            }
                        }
                    ],
                    "pageNumber": 0,
                    "pageSize": 50,
                    "totalSize": 1
                }

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        return self.client.query_records(
            self.collection_id,
            filter=filter,
            order_by=order_by,
            page_number=page_number,
            page_size=page_size
        )