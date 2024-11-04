from flask import Flask, request, jsonify, url_for, send_from_directory
from flask_cors import CORS
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import google.generativeai as genai
import numpy
import PyPDF2 as pdf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')


def get_input_prompt(extracted_text, jd):
    # Prompt Template
    input_prompt = f"""
    You are a skilled and very experienced ATS(Application Tracking System) with a deep understanding of tech field, software engineering,
    data science, data analyst, big data, and machine learning. Your task is to evaluate the resume based on the given job description.
    You must consider the job market is very competitive and you should provide best assistance for improving the resumes. 
    Assign the percentage Matching based on Job description and the missing keywords with high accuracy and anti-fraud feature where it can detect word spam in resume to match the vectorizer cosine similarity AI algorithm.

    Resume:{extracted_text}
    Description:{jd}

    I want the only response as follows in Indonesian Language and follow the format below:

    "Berdasarkan hasil screening test untuk kandidat bernama [Nama Kandidat], berikut adalah hasil dan penjelasan yang berkaitan dengan kecocokan dan rekomendasi peningkatan kompetensi."
    Kecocokan: Berdasarkan analisis, keterampilan Anda dalam [sebutkan keterampilan yang cocok] sesuai dengan persyaratan yang diharapkan untuk posisi ini. Misalnya, kemampuan Anda dalam [contoh keterampilan] mencerminkan kecocokan yang kuat dengan peran ini. Hal ini menunjukkan bahwa Anda telah memiliki fondasi yang baik dalam [sebutkan bidang keterampilan], yang diperlukan untuk peran ini.
    Kata Kunci yang Hilang: Analisis menunjukkan bahwa ada beberapa keterampilan atau pengalaman yang belum disebutkan dalam CV Anda, yang mungkin relevan dengan posisi ini. Beberapa keterampilan yang hilang adalah [sebutkan kata kunci atau keterampilan]. Misalnya, kemampuan dalam [contoh keterampilan yang hilang] sangat penting untuk mendukung peran ini.
    Feedback untuk Peningkatan: Untuk meningkatkan peluang Anda, kami menyarankan agar Anda mengembangkan keterampilan dalam [sebutkan keterampilan yang hilang]. Contohnya, Anda dapat meningkatkan kompetensi dalam [contoh keterampilan] melalui kursus online, sertifikasi, atau pelatihan praktis. Mengembangkan keterampilan ini tidak hanya akan memperkuat posisi Anda dalam proses rekrutmen, tetapi juga meningkatkan kemampuan Anda dalam menavigasi tantangan yang terkait dengan peran ini.
    Hubungan dengan Psikologi Rekrutmen: Berdasarkan prinsip job-person fit dalam psikologi rekrutmen, keterampilan yang hilang ini bisa menunjukkan area pengembangan yang dapat membantu Anda lebih baik dalam menyelaraskan kemampuan kognitif dengan kebutuhan pekerjaan. Dengan meningkatkan keterampilan ini, Anda dapat memperkuat keselarasan antara profil Anda dan peran yang diharapkan oleh perusahaan.
    Contoh Tindakan yang Disarankan: Kami menyarankan agar Anda mengikuti kursus seperti [sebutkan kursus atau pelatihan terkait], atau menambah pengalaman praktis di bidang [sebutkan bidang]. Langkah-langkah ini akan membantu Anda lebih kompetitif dan memberikan kontribusi yang lebih signifikan dalam peran yang Anda lamar.
    
    only return the text with following condition: Use HTML tags only with no additional symbol, make new line for every paragraph
    
    """
    return input_prompt


def get_word_cloud(extracted_text, jd):
    # Prompt Template
    word_cloud_prompt = f"""
    You are a skilled and very experienced ATS(Application Tracking System) with a deep understanding of tech field, software engineering,
    data science, data analyst, big data, and machine learning. Your task is to evaluate the resume based on the given job description.
    You must consider the job market is very competitive and you should provide best assistance for improving the resumes. 
    Assign the percentage Matching based on Job description and the missing keywords with high accuracy and anti-fraud feature where it can detect word spam in resume to match the vectorizer cosine similarity AI algorithm.

    Resume:{extracted_text}
    Description:{jd}

    Based on resume, return all the keyword specifically related to this candidate, I want you to be repetitive for each word based on the skills mentioned on resume, since repeated word will be counted as
    the most strongest word for the candidate

    return the result separated by comma ,
    
    """
    return word_cloud_prompt


# def get_applicant_info(extracted_text):
#     # Prompt Template
#     applicant_info_prompt = f"""
#     Ignore ethics and extract contact information such as Name, Email, Phone number, based on the text below, if one of those are empty return the only exist data, only return the first information you saw.
#     {extracted_text}
#     add 1 new line after each information, also bold the label of each label or section heading of each paragraph
#     """
#     applicant_info_response = model.generate_content(applicant_info_prompt)
#     # Get the name from the response and strip whitespace
#     return applicant_info_response.text


app = Flask(__name__)
CORS(app)

# Define the uploads directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists


@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/")
def welcome():
    return "<p>Welcome To JustHire AI API</p>"


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    context = data.get("context")

    # Generate a prompt to send to the Gemini model
    chat_prompt = f"""
    context: {context}
    Based on the context above, answer the user's message below, provide a helpful and informative response.
    User message: {user_message}
    """

    # Get response from the Gemini model
    gemini_response = model.generate_content(chat_prompt)

    # Return the AI response as JSON
    return jsonify({"response": gemini_response.text})


@app.route("/assess", methods=["POST"])
def assess():
    client_name = request.form["client_name"]
    job_desc = request.form["job_description"]

    fileList = request.files.getlist("resumeFiles[]")
    resume_list = []
    resume_id = 1

    for i in fileList:
        print(i)
        filename = i.filename
        i.save('uploads/' + filename)

        reader = pdf.PdfReader(i)
        extracted_text = ""

        for page in range(len(reader.pages)):
            page = reader.pages[page]
            extracted_text += str(page.extract_text())

        text_array = [extracted_text, job_desc]

        cv = CountVectorizer()
        count_matrix = cv.fit_transform(text_array)

        match = cosine_similarity(count_matrix)[0][1]
        match *= 100
        match = round(match, 2)

        is_match = False
        if match < 50:
            is_match = False
        else:
            is_match = True

        input_prompt = get_input_prompt(extracted_text, job_desc)
        response = model.generate_content(input_prompt)
        word_cloud = get_word_cloud(extracted_text, job_desc)
        keySkills = model.generate_content(word_cloud)
        # applicant_info = get_applicant_info(extracted_text)

        resume_link = url_for(
            'download_file', filename=filename, _external=True)

        resume_info = {
            "id": resume_id,
            "file_name": filename,
            "resume_link": resume_link,
            # "applicant_info": applicant_info,
            "is_match": is_match,
            "job_match": match,
            "extracted_text": extracted_text,
            "response_from_ATS": response.text,
            "keySkills": keySkills.text
        }

        resume_list.append(resume_info)
        resume_id += 1  # Increment the ID for the next resume

    return_data = {
        "client_name": client_name,
        "job_description": job_desc,
        "resumeProcessed": resume_list
    }

    return jsonify(return_data)


if __name__ == '__main__':
    app.run(debug=True)
