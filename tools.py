from langchain_core.tools import tool
from typing import Optional
from datetime import date, datetime
from pymongo import MongoClient
from langchain_core.runnables import RunnableConfig
from bson.objectid import ObjectId
from typing import List, Literal, Optional, Union
from db import vector_store
from langchain_core.documents import Document
import uuid
from db import vector_store


client = MongoClient("mongodb://localhost:27017")
db = client["flight_booking"]

@tool
def lookup_available_tours(
    destination: Optional[str] = None,
    duration_days: Optional[int] = None,
) -> list[dict]:
    """
    Truy vấn các tour đang mở để người dùng có thể đặt.
    Sử dụng khi người dùng hỏi về các tour cụ thể để đi du lịch (có thể đặt).
    Args:
        destination (Optional[str]): Điểm đến du lịch (Ví dụ: Đà Lạt, Phú Quốc, Sa Pa).
        duration_days (Optional[int]): Số ngày của tour (Ví dụ: 1,2,3).
    """
    query = {}
    if destination:
        query["destination"] = destination
    if duration_days:
        query["duration_days"] = duration_days
    print(query)
    results = list(db.tours.find(query))

    return results

@tool
def search_hotels(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
    checkin_date: Optional[str] = None,
    checkout_date: Optional[str] = None,
) -> list[dict]:
    """
    Search for hotels based on location, name, price tier, check-in date, and check-out date.

    Args:
        location (Optional[str]): The location of the hotel. Must include correct Vietnamese accents (e.g., "Đà Lạt", not "Da Lat"). Defaults to None.
        name (Optional[str]): The name of the hotel. Defaults to None.
        price_tier (Optional[str]): The price tier of the hotel. Defaults to None. Examples: mid, luxury, budget
        checkin_date (Optional[str]): The check-in date in 'YYYY-MM-DD' format. Defaults to None.
        checkout_date (Optional[str]): The check-out date in 'YYYY-MM-DD' format. Defaults to None.

    Returns:
        list[dict]: A list of hotel dictionaries matching the search criteria.
    
    Notes: Date format must be in 'YYYY-MM-DD'. For example, "3/5" (spoken or written) should be converted to "2025-05-03".
    """
    query = {}
    if location:
        # user_input = departure_airport.lower()  # Chuyển thành chữ thường để tránh lỗi nhập
        # departure_airport = city_to_airport.get(user_input)
        query["location"] = location
    if name:
        # user_input = arrival_airport.lower()  # Chuyển thành chữ thường để tránh lỗi nhập
        # arrival_airport = city_to_airport.get(user_input)
        query["name"] = name
    if price_tier:
        query["price_tier"] = price_tier
    if checkin_date:
        query["checkin_date"] = checkin_date
    if checkout_date:
        query["checkout_date"] = checkout_date
    query["booked"] = 0
    results = list(db.hotels.find(query))

    return results

@tool
def book_hotel(hotel_id: str, config: RunnableConfig) -> str:
    """
    Book a hotel by its ID.

    Args:
        hotel_id (int): The ID of the hotel to book.

    Returns:
        str: A message indicating whether the hotel was successfully booked or not.
    
    
    Lưu ý: Nếu người dùng đã chọn, hãy tìm hotel_id từ danh sách khách sạn đã trả về trước đó và tự động truyền vào khi gọi book_hotel(hotel_id, config) thay vì bắt người dùng nhập ID.
    Trước khi đặt vé thì hỏi lại xác nhận của người dùng .
    """
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)
    hotel = db['hotels'].find_one({"_id": ObjectId(hotel_id)})

    # Kiểm tra field 'booked'
    if hotel:
        if hotel.get("booked") == 1:
            return f"This hotel {hotel_id} has already been booked."
        else:
            result = db["hotels"].update_one(
                {"_id": ObjectId(hotel_id)},
                {"$set": {"booked": 1}}
            )
            if result.modified_count:
                db.hotel_bookings.insert_one({
                "user_id": ObjectId(user_id),
                "hotel_id": ObjectId(hotel_id),
                "booking_time": datetime.now(),
                "status": "confirmed"
                })
                return f"Hotel {hotel_id} successfully booked."
            else:
                return f"Failed to update booking status for hotel with ID {hotel_id}."
    else:
        return f"No hotel found with ID {hotel_id}."

    

    # Kiểm tra kết quả
    # if result.modified_count:
    #     db.hotel_bookings.insert_one({
    #     "user_id": ObjectId(user_id),
    #     "hotel_id": ObjectId(hotel_id),
    #     "booking_time": datetime.now(),
    #     "status": "confirmed"
    #     })
    #     return f"Hotel {hotel_id} successfully booked."
    # else:
    #     return f"No hotel found with ID {hotel_id}."
    
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
    # city_to_airport = {
    #     "hồ chí minh": "SGN",
    #     "sài gòn": "SGN",
    #     "hà nội": "HAN",
    #     "đà nẵng": "DAD"
    # }
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
        print(departure_day)
        print(type(departure_day ))
        query["$expr"] = {
            "$eq": [
                {"$dateToString": {"format": "%Y-%m-%d", "date": "$departure_time"}},
                departure_day.strftime("%Y-%m-%d")
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
def get_popular_tourist_destinations(query: str) -> str:
    """
    Truy xuất thông tin về các địa điểm du lịch nổi bật ở Việt Nam.
    """
    docs = vector_store.similarity_search(query)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = vector_store.similarity_search(
        query, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id", None)

    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
    )
    vector_store.add_documents([document])
    return memory

