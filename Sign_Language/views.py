import numpy as np
import cv2
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from SignLanguage.sign_language import predict_sign


# Create your views here.

def home(request):
    if request.method == "POST" and 'image' in request.FILES:
        image_file = request.FILES['image']

        np_img = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        result = predict_sign(img)

        return JsonResponse({'result': result})

    return render(request, 'Main.html')
