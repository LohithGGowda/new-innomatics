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