from langchain_core.tools import tool
from typing import Optional
from datetime import date, datetime
from pymongo import MongoClient
from langchain_core.runnables import RunnableConfig
from bson.objectid import ObjectId
from typing import List, Literal, Optional

client = MongoClient("mongodb://localhost:27017")
db = client["flight_booking"]

@tool
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    airline: Optional[str] = None,
    departure_day : Optional[date | datetime] = None,
) -> list[dict]:
    """In ra các chuyến bay dựa vào sân bay khởi hành, sân bay đến, hãng hàng không hoặc
    thời gian khởi hành để người dùng mua vé
    Args:
        departure_airport (Optional[str]): Mã IATA của sân bay khởi hành (VD: "HAN" cho Hà Nội).
        arrival_airport (Optional[str]): Mã IATA của sân bay đến (VD: "SGN" cho Sài Gòn).
        airline (Optional[str]): Hãng hàng không cụ thể cần tìm (VD: "Vietnam Airlines").
        departure_day (Optional[date | datetime]): Ngày khởi hành, định dạng YYYY-MM-DD.
            mặc định là năm 2025.
    """
    city_to_airport = {
        "hồ chí minh": "SGN",
        "sài gòn": "SGN",
        "hà nội": "HAN",
        "đà nẵng": "DAD"
    }
    query = {}
    if departure_airport:
        # user_input = departure_airport.lower()  # Chuyển thành chữ thường để tránh lỗi nhập
        # departure_airport = city_to_airport.get(user_input)
        query["departure_airport"] = departure_airport
    if arrival_airport:
        # user_input = arrival_airport.lower()  # Chuyển thành chữ thường để tránh lỗi nhập
        # arrival_airport = city_to_airport.get(user_input)
        query["arrival_airport"] = arrival_airport
    if airline:
        query["airline"] = airline
    if departure_day:
        print(departure_day )
        print(type(departure_day ))
        query["$expr"] = {
            "$eq": [
                {"$dateToString": {"format": "%Y-%m-%d", "date": "$departure_time"}},
                departure_day .strftime("%Y-%m-%d")
            ]
        }
    print(query)
    results = list(db.flights.find(query))

    return results

@tool
def book_flight(flight_id: str, config: RunnableConfig) -> str:
    """
    Đặt chuyến bay theo ID của chuyến bay.
    Lưu ý: Nếu người dùng đã chọn, hãy tìm flight_id từ danh sách chuyến bay đã trả về trước đó và tự động truyền vào khi gọi book_flight(flight_id, config) thay vì bắt người dùng nhập ID.
    Trước khi đặt vé thì hỏi lại xác nhận của người dùng .
    Args:
        flight_id (str): Mã số chuyến bay cần đặt.

    Returns:
        str: Một thông báo cho biết chuyến bay đã được đặt thành công hay chưa.
    """
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)
    balance_user = db.users.find_one({"_id": ObjectId(user_id)}, {"balance": 1, "_id": 0})['balance']
    price = db.flights.find_one({"_id": ObjectId(flight_id)}, {"price": 1, "_id": 0})['price']
    if balance_user < price:
        return 'Tài khoản của bạn không đủ số dư'
    else:
        new_balance = balance_user - price
        db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"balance": new_balance}})
        db.bookings.insert_one({
            "user_id": ObjectId(user_id),
            "flight_id": ObjectId(flight_id),
            "booking_time": datetime.now(),
            "status": "confirmed"
        })
        return 'Bạn đã đặt vé thành công'

@tool
def search_shuttles(
    from_airport: Optional[str] = None,
    to: Optional[str] = None,
    pickup_datetime : Optional[datetime] = None,
) -> list[dict]:
    """In ra các danh sách xe đưa đón sân bay dựa vào sân bay, điểm đến hoặc
    thời gian đón để người dùng đặt xe.
    lưu ý: tham số truyền vào cần là tiếng việt. Ví dụ:
    from_airport(str): SGN
    to: Quận 1
    """
    query = {}
    if from_airport:
        query["from_airport"] = from_airport
    if to:
        query["to"] = to
    if pickup_datetime:
        print(pickup_datetime )
        print(type(pickup_datetime))
        query["pickup_datetime"] = pickup_datetime
    print(query)
    results = list(db.airport_shuttles.find(query))

    return results

@tool
def book_shuttle(shuttle_id: str, config: RunnableConfig) -> str:
    """
    Đặt shuttle theo ID của shuttle.

    Lưu ý: Nếu người dùng đã chọn từ danh sách trả về trước đó, hãy tìm shuttle_id từ danh sách đó và tự động truyền id vào book_shuttle(shuttle_id, config) thay vì bắt người dùng nhập ID.

    Khi nhận được phản hồi "có" từ người dùng, tự động lấy ID của shuttle đầu tiên trong danh sách đã trả về và truyền vào hàm book_shuttle().

    Args:
        shuttle_id (str): Mã số xe đưa đón cần đặt.

    Returns:
        str: Một thông báo cho biết đã đặt xe thành công hay chưa.
    """
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)
    balance_user = db.users.find_one({"_id": ObjectId(user_id)}, {"balance": 1, "_id": 0})['balance']
    price = db.airport_shuttles.find_one({"_id": ObjectId(shuttle_id)}, {"price": 1, "_id": 0})['price']
    if balance_user < price:
        return 'Tài khoản của bạn không đủ số dư'
    else:
        new_balance = balance_user - price
        db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"balance": new_balance}})
        db.shuttle_bookings.insert_one({
            "user_id": ObjectId(user_id),
            "shuttle_id": ObjectId(shuttle_id),
            "booking_time": datetime.now(),
            "status": "confirmed"
        })
        return 'Bạn đã đặt vé thành công'

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = recall_vector_store.similarity_search(
        query, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]