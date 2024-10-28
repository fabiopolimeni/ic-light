#!/bin/bash

EVENT_ID=$(curl -X POST https://3nu-ic-light.hf.space/call/process_image -s -H "Content-Type: application/json" -d '{
    "data": [
        {"path":"http://0.0.0.0:7860/file=/private/var/folders/p_/0mqf08mx3gn8gj3x45fq7w0w0000gn/T/gradio/b2e78ed566549e8b20ee29d70166a64818bdd7163996977711e028344a62ff1f/7a635dc0-786e-4377-8c66-b478d8cf599d.png"},
        {"path":"http://0.0.0.0:7860/file=/private/var/folders/p_/0mqf08mx3gn8gj3x45fq7w0w0000gn/T/gradio/c1b6a6fefe02946e19fe0d7c28e9b8323ea3619282d2159c9d34ea71b5359aca/dd038598-3c4c-48d3-94f1-72534a8266a7.png"},
        "a van in a construction site with graffitis walls",
        512,
        512,
        1,
        12345,
        10,
        "best quality",
        "lowres, bad anatomy, bad hands, cropped, worst quality",
        2,
        1.5,
        0.8,
        0.5,
        "ENVIRONMENT"
    ]
}' | awk -F'"' '{ print $4}')

curl -N "http://0.0.0.0:7860/call/process_image/$EVENT_ID"
