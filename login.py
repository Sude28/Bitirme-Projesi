import os
import nltk
import re


nltk.download('punkt_tab')
nltk_path = os.path.join(os.getcwd(), "nltk_data")
os.environ["NLTK_DATA"] = nltk_path
nltk.download("punkt", download_dir=nltk_path)


# punkt indirildikten sonra import et
from nltk.tokenize import sent_tokenize

from playwright.sync_api import sync_playwright
import pandas as pd
import time


import spacy

nlp = spacy.load("tech_ner_model_v12")


LINKEDIN_EMAIL = "sudetelli002@hotmail.com"
LINKEDIN_PASSWORD = "280602Sudebuse"

# Anahtar teknoloji kelimeleri
tech_keywords = [
    # Backend & Diller
    "python", "java", "javascript", "typescript", "c#", "c++", "go", "ruby", "php", "rust", ".net", "spring", "spring boot","node.js","node js","nodejs","r",
    # Frontend
    "react", "react.js","reactjs","react js", "vue", "vue.js","vuejs","vue js" ,"angular", "next.js","nextjs","next js" ,"nuxt.js","nuxt js","nuxtjs", "svelte", "html", "html5", "css", "css3", "sass", "less",
    # Mobil
    "flutter", "react native","reactnative", "kotlin", "swift", "android", "ios", "xamarin",
    # DevOps / CI-CD
    "docker", "kubernetes", "helm", "jenkins", "git", "gitlab", "github", "ci", "cd", "ci/cd", "circleci", "travis", "bitbucket", "terraform", "ansible", "puppet","kubeflow",
    # Veri Tabanları
    "postgresql", "mysql", "mariadb", "sqlite", "oracle", "mongodb", "redis", "elasticsearch", "firebase", "realm",
    # Cloud & API
    "aws", "azure", "gcp", "google cloud", "digitalocean", "heroku", "netlify", "vercel", "rest", "restful", "graphql", "api",
    # ML & Veri Bilimi
    "tensorflow", "pytorch", "sklearn", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "openai", "huggingface",
    # NLP
    "nlp", "spacy", "nltk", "bert", "transformer", "gpt", "machine learning", "deep learning", "fasttext",
    # Araçlar
    "jira", "confluence", "trello", "notion", "figma", "xd", "zeplin", "postman", "insomnia",
    # Diğer Frameworkler / Araçlar
    "flask", "django", "fastapi", "express.js", "laravel", "nestjs", "axios", "bootstrap", "tailwind", "material-ui", "chakra-ui",
    # Gözlem / Monitoring
    "prometheus", "grafana", "logstash", "kibana", "new relic", "datadog",
    # Messaging / Stream
    "kafka", "rabbitmq", "mqtt",
    # Diğer
    "opencv", "yolo", "cv2", "tableau", "powerbi", "snowflake","linux","cisco"
]




# Teknoloji cümlelerini ayıklamak için:
def extract_all_tech_sentences(text, keywords):
    from nltk.tokenize import sent_tokenize
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)
    text = re.sub(r"\s+", " ", text)
    sentences = sent_tokenize(text)
    keywords_regex = r'\b(?:' + '|'.join(re.escape(kw.lower()) for kw in keywords) + r')\b'
    # 7. Teknoloji içeren cümleleri topla
    tech_sentences = []
    for sentence in sentences:
        if re.search(keywords_regex, sentence.lower()):
            tech_sentences.append(sentence.strip())

    return " ".join(tech_sentences) if tech_sentences else ""


# 📌 Açıklama alma fonksiyonu
def get_job_description(page):
    try:
        # En dış kapsayıcıdan tüm metni düz olarak al
        raw_text = page.locator("div.jobs-box__html-content").inner_text().strip()

        # Satır sonlarını normalize et
        lines = raw_text.split("\n")
        clean_lines = [line.strip() for line in lines if line.strip()]

        # Aynı cümleyi/ifadeyi iki kez eklememek için benzersizleştir
        unique_lines = list(dict.fromkeys(clean_lines))

        return "\n".join(unique_lines)

    except Exception as e:
        print(f"⚠️ Açıklama alınamadı: {e}")
        return ""




def scroll_and_load_jobs(page, max_jobs=10, max_wait_scrolls=10):
    print(" Scroll & Sayfa Geçişi Başlatılıyor...")
    previous_count = -1
    no_change_count = 0

    for _ in range(max_wait_scrolls):
        page.evaluate("document.querySelector('div.scaffold-layout__list').scrollBy(0, 3000)")
        time.sleep(2)

        job_cards = page.locator("li.ember-view.scaffold-layout__list-item")
        current_count = job_cards.count()

        if current_count == previous_count:
            no_change_count += 1
            print("Yeni ilan yüklenmedi.")
        else:
            no_change_count = 0
            print(f" Yeni ilan yüklendi. Toplam ilan: {current_count}")

        previous_count = current_count

        if current_count >= max_jobs:
            print(f" {max_jobs} ilana ulaşıldı.")
            break

        if no_change_count >= 5:
            print(" Scroll ile yeni ilan uzun süre gelmedi. Duruluyor.")
            break

    print(" Scroll tamamlandı.")










# Pozisyonları tanımla
positions = {
    "Backend Developer": {
        "url": "https://www.linkedin.com/jobs/search?keywords=Backend%20Developer&location=Türkiye",
        "filter_keywords": ["backend", "node.js", "javascript", "python", "java", ".net","back end"]
    },
    "Frontend Developer": {
        "url": "https://www.linkedin.com/jobs/search?keywords=Frontend%20Developer&location=Türkiye",
        "filter_keywords": ["frontend", "front end", "front - end", "front-end", "web", "angular", "react.js", "html", "css", "php", "ön uç geliştirici"]
    },
    "Mobile Developer": {
        "url": "https://www.linkedin.com/jobs/search?keywords=Mobile%20Developer&location=Türkiye",
        "filter_keywords": ["android", "ios", "flutter", "swift", "kotlin", "react native", "mobile", "mobil"]
    },
    "Full Stack Developer": {
        "url": "https://www.linkedin.com/jobs/search?keywords=Full%20Stack%20Developer&location=Türkiye",
        "filter_keywords": ["fullstack", "full stack", "full-stack", "full - stack"]
    },
    "Data Scientist": {
        "url": "https://www.linkedin.com/jobs/search?keywords=Data%20Scientist&location=Türkiye",
        "filter_keywords": ["veri", "data", "data science", "data scientist", "data analysis"]
    },
    "DevOps": {
        "url": "https://www.linkedin.com/jobs/search?keywords=DevOps&location=Türkiye",
        "filter_keywords": ["devops"]
    },
    "Siber Güvenlik Uzmanı": {
        "url": "https://www.linkedin.com/jobs/search?keywords=Siber%20Güvenlik%20Uzmanı&location=Türkiye",
       "filter_keywords": ["siber", "penetrasyon", "sızma", "test", "güvenlik", "security", "cyber"]
    },
    "Network Yönetimi": {
       "url": "https://www.linkedin.com/jobs/search?keywords=Network%20Yönetimi&location=Türkiye",
       "filter_keywords": ["network", "sistem", "ağ", "system"]
    },
    "Software Engineer": {
        "url": "https://www.linkedin.com/jobs/search?keywords=Software%20Engineer&location=Türkiye",
        "filter_keywords": ["software engineer", "yazılım mühendisi", "software developer", "yazılım uzmanı", "yazılım"]
    },
    "Yapay Zeka Mühendisi": {
       "url": "https://www.linkedin.com/jobs/search/?keywords=yapay%20zeka%20mühendisi&location=Türkiye",
      "filter_keywords": ["yapay zeka", "ai"]
    }


}


def is_title_relevant(title, keywords):
    title_lower = title.lower()
    return any(kw.lower() in title_lower for kw in keywords)




# 📌 Ana scraping fonksiyonu
all_results = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    # Giriş yap
    page.goto("https://www.linkedin.com/login")
    page.fill("input#username", LINKEDIN_EMAIL)
    page.fill("input#password", LINKEDIN_PASSWORD)
    page.click("button[type='submit']")
    page.wait_for_url("https://www.linkedin.com/feed/", timeout=10000)
    time.sleep(5)

    for position, data in positions.items():
        print(f"\n🔍 {position} pozisyonu için ilanlar çekiliyor...")
        page.goto(data["url"])
        time.sleep(3)

        try:
            page.wait_for_selector("div.job-card-container", timeout=10000)
        except:
            print(f"⚠️ Uyarı: {position} için ilanlar DOM'a gelmedi, atlanıyor.")
            continue
        scroll_and_load_jobs(page, max_jobs=10)

        collected = 0
        max_ads = 10
        current_page = 1



        while collected < max_ads:
            print(f"\n📄 Sayfa {current_page} işleniyor...")
            scroll_and_load_jobs(page, max_jobs=10)

            index = 0
            job_cards = page.locator("div.job-card-container")

            while index < job_cards.count() and collected < max_ads:
                print(f"🔢 {index + 1}. ilan işleniyor (Toplam görünür kart: {job_cards.count()})")
                card = job_cards.nth(index)

                try:
                    card.wait_for(state="visible", timeout=10000)
                    card.scroll_into_view_if_needed()
                    page.mouse.wheel(0, 1000)
                    page.wait_for_timeout(2000)
                    card.click()
                    page.wait_for_timeout(2000)

                    title = page.locator("h1.t-24.t-bold.inline").inner_text(timeout=3000)
                    if not is_title_relevant(title, data["filter_keywords"]):
                        print(f" Atlandı (uygunsuz başlık): {title}")
                    else:
                        print(f" {index + 1}. ilan uygun, veri alınıyor.")
                        # ... verileri al, NLP işlemleri yap ...

                        company = page.locator("div.job-details-jobs-unified-top-card__company-name a").inner_text(
                            timeout=3000)
                        location = page.locator(
                            "div[class*='job-details-jobs-unified-top-card'] span[class^='tvm__text']").nth(
                            0).inner_text(
                            timeout=5000)
                        date = page.locator(
                            "div[class*='job-details-jobs-unified-top-card'] span[class^='tvm__text']").nth(
                            2).inner_text(timeout=5000)

                        description = get_job_description(page)
                        short_description = extract_all_tech_sentences(description, tech_keywords)
                        doc = nlp(short_description)
                        technologies = [ent.text.lower().strip() for ent in doc.ents if ent.label_ == "TECH"]

                        # Filtreleme
                        blacklist = {"zorunlu", "tercih", "ihtiyaç", "bilgi", "deneyim", "yeterlilik", "süreç",
                                     "çalışma",
                                     "Computer Science", "Bachelors", "Masters", "developer", "Developer", "adayların",
                                     "aday", "firma", "optimize"}
                        compound_techs = [
                            ("spring", "boot"), ("react", "native"), ("node", "js"), ("vue", "js"),
                            ("next", "js"), ("google", "cloud"), ("machine", "learning"), ("deep", "learning"),
                            ("rest", "api"), ("fast", "api"), ("power", "bi"), ("visual", "studio")
                        ]

                        technologies = [tech for tech in technologies if tech not in blacklist]

                        merged = []
                        skip = False
                        for i in range(len(technologies)):
                            if skip:
                                skip = False
                                continue
                            if i < len(technologies) - 1 and (technologies[i], technologies[i + 1]) in compound_techs:
                                merged.append(f"{technologies[i]}_{technologies[i + 1]}")
                                skip = True
                            else:
                                merged.append(technologies[i])

                        technologies_cleaned = ", ".join(merged)

                        all_results.append({
                            "position": position,
                            "title": title,
                            "company": company,
                            "location": location,
                            "date": date,
                            "description": description,
                            "short_description": short_description,
                            "technologies": ", ".join(technologies),
                            "technologies_cleaned_v2": technologies_cleaned
                        })

                        collected += 1

                except Exception as e:
                    print(f"⚠️ Hata (ilan {index + 1}): {e}")

                index += 1

            # 📌 Sayfa geçişi denemesi her sayfa sonunda yapılmalı
            try:
                print("🔍 Sayfa geçişi kontrol ediliyor...")
                next_button = page.locator('button[aria-label="View next page"]')

                if next_button.count() > 0 and next_button.is_visible() and next_button.is_enabled():
                    print("➡️ Sonraki sayfaya geçiliyor...\n")
                    next_button.scroll_into_view_if_needed()
                    next_button.click()
                    current_page += 1
                    page.wait_for_selector("div.job-card-container", timeout=10000)
                    time.sleep(3)
                else:
                    print("⛔ Sonraki sayfa butonu görünmüyor. Döngü durduruluyor.")
                    break

            except Exception as e:
                print(f"❌ Sayfa geçişi sırasında hata oluştu: {e}")
                break

        browser.close()

# 💾 Excel'e yaz
df = pd.DataFrame(all_results)
df.to_excel("backend.xlsx", index=False)
print("\n💾 Tüm ilanlar kaydedildi: backend.xlsx")