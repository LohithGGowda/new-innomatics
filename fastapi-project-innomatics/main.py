# 🎓 Online Course Platform
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI()

# Create main.py. Add GET / returning {\'message\': \'Welcome to LearnHub Online Courses\'}.

@app.get("/")
def get_message():
    return {"Welcome to Loki's LearnHub Online Courses"}

# Create a courses list with at least 6 courses: id, title, instructor, category (Web Dev/Data Science/Design/DevOps), level (Beginner/Intermediate/Advanced), price (int), seats_left (int).

courses = [
    {"title": "Data_science", "instructor": "Ranjini_P_S", "category": "Science", "level": "Advanced", "price": 999 , "seats_left": 16, },
    {"title": "web_development", "instructor": "suresh_K", "category": "Web Dev", "level": "Beginner", "price": 0 , "seats_left": 20, },
    {"title": "data_science", "instructor": "Anjali_R", "category": "Data Science", "level": "Intermediate", "price": 499 , "seats_left": 10, },
    {"title": "devops_fundamentals", "instructor": "Vikram_T", "category": "DevOps", "level": "Intermediate", "price": 299 , "seats_left": 8, },
    {"title": "machine_learning", "instructor": "Neha_M", "category": "Data Science", "level": "Advanced", "price": 799 , "seats_left": 12, },
    {"title": "ui_ux_design", "instructor": "Amit_K", "category": "Design", "level": "Intermediate", "price": 299 , "seats_left": 7, },
    {"title": "cloud_computing", "instructor": "Lohith", "category": "DevOps", "level": "Beginner", "price": 0 , "seats_left": 25, }
]

# Build GET /courses returning all courses, total, and total_seats_available.

@app.get("/courses")
def get_courses(): 
    return {
        "courses": courses,
        "total": len(courses),
        "total_seats_available": sum(course["seats_left"] for course in courses)
    }

# Build GET /courses/{course_id}. Return the course or an error. Test both cases.

@app.get("/courses/{title_in}")
def get_course_byid(title_in: str):
    
    filtered_course = []

    for course in courses:
        if course["title"].lower() == title_in.lower():
            filtered_course.append(course)

    if len(filtered_course) == 0:
        return {"error": "No courses found by this title"}

    return {
        "courses": filtered_course
    }

# Create enrollments = [] and enrollment_counter = 1. Build GET /enrollments returning all enrollments and total.

enrollments = []
enrollment_counter = 1

@app.get("/enrollments")
def get_enrollments():
    return {
        "enrollments": enrollments,
        "total": len(enrollments)
    }

# Build GET /courses/summary (above /courses/{course_id}). Return: total courses,
# free courses count (price=0), most expensive course, total seats across all courses, and a count by category.

@app.get("/courses/summary")
def get_courses_summary():
    try:
        total_courses = len(courses)
        free_courses_count = sum(1 for course in courses if course["price"] == 0)
        most_expensive_course = max(courses, key=lambda x: x["price"])
        total_seats = sum(course["seats_left"] for course in courses)
        count_by_category = {category: sum(1 for course in courses if course["category"] == category) for category in set(course["category"] for course in courses)}

        return {
            "total_courses": total_courses,
            "free_courses_count": free_courses_count,
            "most_expensive_course": most_expensive_course,
            "total_seats": total_seats,
            "count_by_category": count_by_category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create EnrollRequest: student_name (min 2), course_id (gt 0), email (str, min_length=5),
# payment_method (default 'card'), coupon_code (default ''), gift_enrollment (bool), recipient_name (str).

class EnrollRequest(BaseModel):
    student_name: str = Field(min_length=2)
    course_id: int = Field(gt=0)
    email: str = Field(min_length=5)
    payment_method: str = "card"
    coupon_code: str = ""
    gift_enrollment: bool = False
    recipient_name: str = ""


# Write find_course(course_id) helper. Return the course if found, else None.

def find_course(course_id):
    for course in courses:
        if course.get("id") == course_id:
            return course
    return None


# Write calculate_enrollment_fee(price, seats_left, coupon_code).
# Apply 10% early-bird if seats_left > 5, then apply coupons: STUDENT20 (20%), FLAT500 (₹500 off).

def calculate_enrollment_fee(price, seats_left, coupon_code):

    discount = 0
    final_price = price

    if seats_left > 5:
        early_discount = 0.1 * price
        final_price -= early_discount
        discount += early_discount

    if coupon_code == "STUDENT20":
        coupon_discount = 0.2 * final_price
        final_price -= coupon_discount
        discount += coupon_discount

    elif coupon_code == "FLAT500":
        final_price -= 500
        discount += 500

    return {
        "final_fee": max(final_price, 0),
        "total_discount": discount
    }


# Build POST /enrollments. Check course exists, check seats_left > 0, apply fee calculation,
# reduce seats_left, support gift enrollment, and return enrollment details.

@app.post("/enrollments")
def enroll_course(request: EnrollRequest):

    global enrollment_counter

    if request.gift_enrollment and request.recipient_name == "":
        return {"error": "Recipient name required for gift enrollment"}

    course = find_course(request.course_id)

    if not course:
        return {"error": "Course not found"}

    if course["seats_left"] <= 0:
        return {"error": "No seats available"}

    fee_data = calculate_enrollment_fee(
        course["price"],
        course["seats_left"],
        request.coupon_code
    )

    course["seats_left"] -= 1

    enrollment = {
        "enrollment_id": enrollment_counter,
        "student_name": request.student_name,
        "recipient_name": request.recipient_name if request.gift_enrollment else request.student_name,
        "course_title": course["title"],
        "instructor": course["instructor"],
        "original_price": course["price"],
        "discount": fee_data["total_discount"],
        "final_fee": fee_data["final_fee"]
    }

    enrollments.append(enrollment)
    enrollment_counter += 1

    return enrollment


# Write filter_courses_logic() helper. Filter using optional params: category, level, max_price, has_seats.

def filter_courses_logic(category=None, level=None, max_price=None, has_seats=None):

    filtered = []

    for course in courses:

        if category is not None and course["category"].lower() != category.lower():
            continue

        if level is not None and course["level"].lower() != level.lower():
            continue

        if max_price is not None and course["price"] > max_price:
            continue

        if has_seats is not None:
            if has_seats and course["seats_left"] <= 0:
                continue
            if not has_seats and course["seats_left"] > 0:
                continue

        filtered.append(course)

    return filtered


# Build GET /courses/filter with optional params: category, level, max_price, has_seats.
# Return filtered courses and count.

from typing import Optional
from fastapi import Query

@app.get("/courses/filter")
def filter_courses(
    category: Optional[str] = Query(default=None),
    level: Optional[str] = Query(default=None),
    max_price: Optional[int] = Query(default=None),
    has_seats: Optional[bool] = Query(default=None)
):

    filtered = filter_courses_logic(category, level, max_price, has_seats)

    return {
        "count": len(filtered),
        "courses": filtered
    }

# Create NewCourse model: title, instructor, category, level (min 2), price (>=0), seats_left (>0).

class NewCourse(BaseModel):
    title: str = Field(min_length=2)
    instructor: str = Field(min_length=2)
    category: str = Field(min_length=2)
    level: str = Field(min_length=2)
    price: int = Field(ge=0)
    seats_left: int = Field(gt=0)


# Build POST /courses — reject duplicate titles and return created course.

@app.post("/courses")
def create_course(course: NewCourse):

    for c in courses:
        if c["title"].lower() == course.title.lower():
            return {"error": "Course with this title already exists"}

    new_course = {
        "id": len(courses) + 1,
        "title": course.title,
        "instructor": course.instructor,
        "category": course.category,
        "level": course.level,
        "price": course.price,
        "seats_left": course.seats_left
    }

    courses.append(new_course)

    return new_course


# Build PUT /courses/{course_id} — update price and/or seats_left if provided.

from typing import Optional

@app.put("/courses/{course_id}")
def update_course(course_id: int, price: Optional[int] = None, seats_left: Optional[int] = None):

    course = find_course(course_id)

    if not course:
        return {"error": "Course not found"}

    if price is not None:
        course["price"] = price

    if seats_left is not None:
        course["seats_left"] = seats_left

    return course


# Build DELETE /courses/{course_id} — prevent deletion if enrollments exist.

@app.delete("/courses/{course_id}")
def delete_course(course_id: int):

    course = find_course(course_id)

    if not course:
        return {"error": "Course not found"}

    for e in enrollments:
        if e["course_title"] == course["title"]:
            return {"error": "Cannot delete course with enrolled students"}

    courses.remove(course)

    return {"message": "Course deleted successfully"}


# Create wishlist = []. Build POST /wishlist/add — avoid duplicate student+course entries.

wishlist = []

@app.post("/wishlist/add")
def add_to_wishlist(student_name: str, course_id: int):

    course = find_course(course_id)

    if not course:
        return {"error": "Course not found"}

    for item in wishlist:
        if item["student_name"] == student_name and item["course_id"] == course_id:
            return {"error": "Already in wishlist"}

    wishlist.append({
        "student_name": student_name,
        "course_id": course_id,
        "price": course["price"]
    })

    return {"message": "Added to wishlist"}


# Build GET /wishlist — return wishlist items and total value.

@app.get("/wishlist")
def get_wishlist():

    total_value = sum(item["price"] for item in wishlist)

    return {
        "wishlist": wishlist,
        "total_value": total_value
    }


# Build DELETE /wishlist/remove/{course_id} — remove item for a specific student.

@app.delete("/wishlist/remove/{course_id}")
def remove_from_wishlist(course_id: int, student_name: str):

    for item in wishlist:
        if item["course_id"] == course_id and item["student_name"] == student_name:
            wishlist.remove(item)
            return {"message": "Removed from wishlist"}

    return {"error": "Item not found"}


# Build POST /wishlist/enroll-all — enroll all wishlist courses for a student.

@app.post("/wishlist/enroll-all")
def enroll_all(student_name: str, payment_method: str):

    global enrollment_counter

    enrolled = []
    total_fee = 0

    for item in wishlist[:]:
        if item["student_name"] == student_name:

            course = find_course(item["course_id"])

            if course and course["seats_left"] > 0:

                fee_data = calculate_enrollment_fee(course["price"], course["seats_left"], "")

                course["seats_left"] -= 1

                enrollment = {
                    "enrollment_id": enrollment_counter,
                    "student_name": student_name,
                    "course_title": course["title"],
                    "final_fee": fee_data["final_fee"]
                }

                enrollments.append(enrollment)
                enrolled.append(enrollment)
                total_fee += fee_data["final_fee"]

                enrollment_counter += 1
                wishlist.remove(item)

    return {
        "total_enrolled": len(enrolled),
        "total_fee": total_fee,
        "enrollments": enrolled
    }


# Build GET /courses/search — search keyword in title, instructor, category.

@app.get("/courses/search")
def search_courses(keyword: str):

    results = []

    for course in courses:
        if keyword.lower() in course["title"].lower() or \
           keyword.lower() in course["instructor"].lower() or \
           keyword.lower() in course["category"].lower():
            results.append(course)

    return {
        "total_found": len(results),
        "courses": results
    }


# Build GET /courses/sort — sort by price/title/seats_left with optional order.

@app.get("/courses/sort")
def sort_courses(sort_by: str = "price", order: str = "asc"):

    if sort_by not in ["price", "title", "seats_left"]:
        return {"error": "Invalid sort field"}

    reverse = True if order == "desc" else False

    sorted_courses = sorted(courses, key=lambda x: x[sort_by], reverse=reverse)

    return {
        "sorted_by": sort_by,
        "order": order,
        "courses": sorted_courses
    }


# Build GET /courses/page — implement pagination with page and limit.

@app.get("/courses/page")
def paginate_courses(page: int = 1, limit: int = 3):

    start = (page - 1) * limit
    end = start + limit

    total_pages = (len(courses) + limit - 1) // limit

    return {
        "page": page,
        "total_pages": total_pages,
        "courses": courses[start:end]
    }


# Build GET /enrollments/search — search enrollments by student_name.

@app.get("/enrollments/search")
def search_enrollments(student_name: str):

    results = [e for e in enrollments if student_name.lower() in e["student_name"].lower()]

    return {
        "total_found": len(results),
        "enrollments": results
    }


# Build GET /enrollments/sort — sort enrollments by final_fee.

@app.get("/enrollments/sort")
def sort_enrollments(order: str = "asc"):

    reverse = True if order == "desc" else False

    sorted_data = sorted(enrollments, key=lambda x: x.get("final_fee", 0), reverse=reverse)

    return {
        "order": order,
        "enrollments": sorted_data
    }


# Build GET /enrollments/page — paginate enrollments.

@app.get("/enrollments/page")
def paginate_enrollments(page: int = 1, limit: int = 3):

    start = (page - 1) * limit
    end = start + limit

    total_pages = (len(enrollments) + limit - 1) // limit

    return {
        "page": page,
        "total_pages": total_pages,
        "enrollments": enrollments[start:end]
    }


# Build GET /courses/browse — combine search, filter, sort, and pagination.

@app.get("/courses/browse")
def browse_courses(
    keyword: Optional[str] = None,
    category: Optional[str] = None,
    level: Optional[str] = None,
    max_price: Optional[int] = None,
    sort_by: str = "price",
    order: str = "asc",
    page: int = 1,
    limit: int = 3
):

    result = courses

    if keyword:
        result = [c for c in result if keyword.lower() in c["title"].lower()]

    if category:
        result = [c for c in result if c["category"].lower() == category.lower()]

    if level:
        result = [c for c in result if c["level"].lower() == level.lower()]

    if max_price is not None:
        result = [c for c in result if c["price"] <= max_price]

    reverse = True if order == "desc" else False

    if sort_by in ["price", "title", "seats_left"]:
        result = sorted(result, key=lambda x: x[sort_by], reverse=reverse)

    start = (page - 1) * limit
    end = start + limit

    return {
        "total": len(result),
        "page": page,
        "courses": result[start:end]
    }

    # total_courses = len(courses)
    # value_of_courses = (value_course for course in courses )
    # free_courses = []
    # if course['price']==0 :
    #     free_courses.append(course)
    # expensive_courses = ex_course if max(value_of_courses) return course else None



    # return {
    #     "total_courses": total_courses,
    #     "free_courses_count": len(free_courses),
    #     "most_expensive_course": expensive_courses,
    #     "total_seats": sum(course["seats_left"] for course in courses),
    #     "count_by_category": {category: sum(1 for course in courses if course["category"] == category) for category in set(course["category"] for course in courses)}
    # }