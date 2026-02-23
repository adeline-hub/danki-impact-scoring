<p align="center">

&nbsp; <a href="https://dankistudio.com">

&nbsp;   <img src="report/assets/logo.png" width="140">

&nbsp; </a>

</p>



<h1 align="center">IDRIS — Impact \& Due Diligence Risk Intelligence Scoring

</h1>

<p align="center">

&nbsp; A data-driven framework for estimating Impact Scoring<br/>

&nbsp; based on 2000 business cases trained by Machine Learning.

</p>



<p align="center">

&nbsp; <a href="https://adeline-hub.github.io/idris/">

&nbsp;   <img src="https://img.shields.io/badge/Report-Live-blue?logo=quarto" alt="Report Live"/>

&nbsp; </a>

&nbsp; <a href="https://adeline-hub.github.io/idris/app.html">

&nbsp;   <img src="https://img.shields.io/badge/Calculator-Live-orange?logo=leaflet" alt="Calculator Live"/>

&nbsp; </a>

&nbsp; <a href="LICENSE">

&nbsp;   <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT"/>

&nbsp; </a>

</p>



---



\## Overview





---



\## Deliverables



---



\## Investment Decision Support Tool





---



\## Strategic Impact





---



\## Why Machine Learning?





---



\## Project Structure



```

idris/

├── data/

│   └── processed/

│       └── investments.parquet

├── docs/                        

│   └── assets/

│       ├── logo.png

│       └── favicon.io

├── notebooks/

│   └── eda\_marimo.py

├── report/

│   ├── index.qmd                

│   ├── app.qmd                  

│   └── assets/

│       ├── logo.png

│       └── favicon.io

├── src/

│   ├── generate\_data.py

│   └── viz.py

├── requirements.txt

└── \_quarto.yml

```



---



\## Local Development (Windows PowerShell)



\### 1. Clone Repository



```powershell

git clone https://github.com/adeline-hub/idris.git

cd idris

```



\### 2. Create and Activate Virtual Environment



```powershell

\# Windows

python -m venv .venv

.\\.venv\\Scripts\\Activate



\# macOS / Linux

python3 -m venv .venv

source .venv/bin/activate

```



\### 3. Install Dependencies



```powershell

pip install -r requirements.txt

```



If generating PDF for the first time:



```powershell

quarto install tinytex

```



\### 4. Generate Dataset



```powershell

python src/generate\_data.py

```



Output: `data/processed/investments.parquet` \[deprecated]



\### 5. Render HTML Report + Calculator



```powershell

cd report

quarto render index.qmd --to html --output-dir ../docs

quarto render app.qmd --to html --output-dir ../docs

cd ..

```



Open locally: `docs/index.html` and `docs/app.html`



\### 6. Render PDF Report



```powershell

cd report

quarto render index.qmd --to pdf

cd ..

```



Output: `report/index.pdf`



\### 7. Publish to GitHub Pages



```powershell

git add .

git commit -m "Update ..."

git push origin main

```



Or use Quarto's built-in publish command:



```powershell

quarto publish gh-pages

```



---



\## References



1\. European commission

2\. AMF

3\. 

4\. 



---



<p align="center">

&nbsp; Built by <a href="https://dankistudio.com"><strong>Danki Studio</strong></a> · Nambona Adeline YANGUERE

</p>



