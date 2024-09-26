from django.shortcuts import render
from dotenv import load_dotenv, find_dotenv
import json
import os
import numpy as np
from movie.models import Movie

import google.generativeai as genai

def recommend(request):
    _ = load_dotenv('api_keys.env')
    genai.configure(api_key=os.getenv('gemini_api_key'))

    def get_embedding_gemini(txt):
        response_emb = genai.embed_content(
            model="models/embedding-001",
            content=txt,
            task_type="retrieval_document",
            title="Embedding of single string")
        return response_emb['embedding']

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    with open('movie_descriptions.json', 'r') as file:
        file_content = file.read()
        movies = json.loads(file_content)
    
    # EJECUTAR SOLO LA PRIMERA VEZ
    # movies_gemini = [movies[i] for i in range(len(movies))]
    # for i in range(len(movies_gemini)):
    #     emb = get_embedding_gemini(movies_gemini[i]['description'])
    #     movies_gemini[i]['embedding'] = emb
    # with open('movie_descriptions_embeddings.json', 'w') as file:
    #     json.dump(movies_gemini, file)
    
    # EJECUTAR DESPUÉS DE LA PRIMERA VEZ
    with open('movie_descriptions_embeddings.json', 'r') as file:
        file_content = file.read()
        movies_gemini = json.loads(file_content)
    
    req = request.GET.get('searchRecommendation')
    if not req:
        return render(request, 'recommendations.html')
    else:
        emb = get_embedding_gemini(req)
        sim = []
        for i in range(len(movies_gemini)):
            sim.append(cosine_similarity(emb,movies_gemini[i]['embedding']))
        sim = np.array(sim)
        idx = np.argmax(sim)
        result = movies[idx]['title']
        movies = Movie.objects.filter(title__icontains=result)
        return render(request, 'recommendations.html', {'searchRecommendation':req, 'movies':movies})