#!/bin/bash -xue
curl 'https://unpkg.com/vue@3.3.10/dist/vue.esm-browser.js' > vue.js
exec uvicorn server:app --reload --host 0
