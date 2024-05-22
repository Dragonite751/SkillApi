
    # input_text = "I have experience with Python, JavaScript, and Machine Learning Algorithms."

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
load_dotenv()
app = FastAPI()

known_skills = [
    "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "PHP", "Swift", "Kotlin", 
    "HTML", "CSS", "React", "Angular", "NodeJS", "ExpressJS", "SQL", "MySQL", 
    "PostgreSQL", "MongoDB", "Redis", "Django", "Flask", "Ruby on Rails", "Spring Boot", 
    ".NET Core", "TensorFlow", "PyTorch", "Keras", "Scikit-Learn", "Natural Language Processing", 
    "NLP", "Computer Vision", "Reinforcement Learning", "Deep Learning", 
    "Data Engineering", "Big Data Analytics", "Data Warehousing", "ETL (Extract, Transform, Load)", 
    "AWS", "Azure", "Google Cloud", "Kubernetes", "Docker", "Jenkins", "Ansible", "Terraform", 
    "React", "Vue.js", "Redux", "VueX", "HTML5", "CSS3", "Responsive Web Design", "Kotlin", 
    "Java", "iOS", "Swift", "Selenium", "JUnit", "TestNG", "Linux", "Unix", "Windows", 
    "Tableau", "Power BI", "Matplotlib", "Seaborn", "Data Analytics", "Business Intelligence", 
    "Communication", "Problem Solving", "Teamwork", "Leadership", "Adaptability", "AutoCAD", 
    "SolidWorks", "CATIA", "MATLAB", "ANSYS", "Thermodynamics", "Fluid Mechanics", 
    "Mechanical Design", "Finite Element Analysis", "CAD", "CAM", "Heat Transfer", 
    "Manufacturing Processes", "Machine Design", "Structural Analysis", "Control Systems", 
    "Robotics", "Materials Science", "Mechatronics", "HVAC", "CNC Machining", "Welding", 
    "Machining", "Product Design", "MERN", "Machine Learning"
]

class JobInfo(BaseModel):
    deg: str
    city: str
    role: str
    country: str
    restext: str

def extract_skills(text):
    lower_case_text = text.lower()
    matched_skills = [skill for skill in known_skills if skill.lower() in lower_case_text]
    matched_skills = set(matched_skills)
    return list(matched_skills)

@app.post("/recommend_jobs/")
def jobrec(info: JobInfo):
    print("hello")
    extracted_main = extract_skills(info.restext)
    role = info.role
    city = info.city
    deg = info.deg
    country = info.country
    url = "https://jsearch.p.rapidapi.com/search"
    querystring = {"query": "{} in {}, {}".format(role, city, country), "page": "1", "num_pages": "10",
                   "employment_types": "{}".format(deg)}
    headers = {
        "X-RapidAPI-Key": "a312d09bf1msh733c7a2bdd51d33p1e0d88jsn7dd4057091d7",
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    print(response)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching job data")
    
    job_data = response.json()['data']

    skillmat = {}
    for i in range(0, len(response.json()['data'])):
        skillmat[i] = response.json()['data'][i]['job_description']

    job_desc = [extract_skills(skillmat[i]) for i in range(len(skillmat))]
    index = list(range(len(skillmat)))

    dictionar = {
        'index': index,
        'job_desc': job_desc
    }
    df = pd.DataFrame.from_dict(dictionar)

    new_row = {
        'index': [len(df)],
        'job_desc': [extracted_main]
    }
    new_row = pd.DataFrame.from_dict(new_row)
    df = pd.concat([df, new_row], ignore_index=True)
    df['job_desc'] = [' '.join(desc) for desc in df['job_desc']]
    
    cv = CountVectorizer(max_features=50, stop_words='english')
    vec = cv.fit_transform(df['job_desc'])
    similarity = cosine_similarity(vec)
    
    def recommend():
        distances = similarity[len(df) - 1]
        job_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
        final_list = [df.iloc[i[0]]['index'] for i in job_list][1:]
        return [job_data[i] for i in final_list]

    final_list = recommend()
    distances = similarity[len(df) - 1]
    job_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:]
    per = [float(job[1]) for job in job_list]
    
    for i in range(len(per)):
        if per[i] == 0.0:
            per[i] = "No particular skills required"
   
    data = {
        "ranks": per,
        "job_info": final_list
    }
    
    return data

# Run the FastAPI server with Uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

       
    