import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, render_template, request
import flask

app = Flask(__name__)

@app.route('/')
def welcome():
    return flask.render_template("welcome.html")

def get_recommendations(title, cosine_sim, indices):
    # Get the index of the job that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores of all jobs with that job
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the jobs based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar jobs
    sim_scores = sim_scores[0:10]

    # Get the job indices
    job_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar jobs
    #
    return job_indices
    # return metadata['cleaned_job_title'].iloc[job_indices]


def main_func(job_name, skill):
    metadata = pd.read_csv('cleaned_concat_new_df.csv')
#    job_name = input('enter job title:')
#    skill = input('enter skills(comma separated):') 

    tfidf = TfidfVectorizer(stop_words='english')
    metadata['description'] = metadata['description'].fillna('')
    metadata['skills'] = metadata['skills'].fillna('')

    #xx = list(metadata['description'])
    #xx.append(job_name)
    tfidf_matrix = tfidf.fit_transform(metadata['description'])
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    #Construct a reverse map of indices and job titles
    data = metadata.drop_duplicates(subset='cleaned_job_title')

    #metadata = metadata.reset_index(drop=True)
    indices = pd.Series(data.index, index=data['cleaned_job_title'])
    ind = get_recommendations(job_name, cosine_sim, indices)
    data2 = metadata.to_numpy()
    jobs = []
    for i in range(10):
        jobs.append(data2[ind[i]])
    df=pd.DataFrame(jobs,columns=['cleaned_job_title','skills', 'description', 'location', 'country', 'industry'])
    #df = df.to_numpy()
    zz = list(df['skills'])
    zz.append(skill)

    tfidf_matrix2 = tfidf.fit_transform(zz)
    cosine_sim2 = linear_kernel(tfidf_matrix2, tfidf_matrix2)
    print(cosine_sim2)
    q_or_n = []
    df = df.to_numpy()

    for i in range(10):
        if cosine_sim2[10,i] > 0.5:
            q_or_n.append('qualified')
        else:
            q_or_n.append('not qualified')

    res = []
    for i in range (10):
        res.append('job name: ' + df[i][0] +  '| qualification: ' + q_or_n[i])
        #print('job name: ' + df[i][0] +  '| qualification: ' + q_or_n[i])
    return res
    

@app.route('/result',methods = ['POST'])
def result():
    to_predict_list = request.form.to_dict()
    to_predict_list=list(to_predict_list.values())
    print(to_predict_list)
    job = to_predict_list[0]
    skills = to_predict_list[1]
    print(job)
    print(skills)
    prediction  = main_func(job, skills)
    return render_template("result.html",prediction=prediction)




#files = ['C:\\Users\Amro\Desktop\Inetworks_internship\templates\welcome.html']

if __name__ == '__main__':
    app.run(debug=True, port = 5000)
