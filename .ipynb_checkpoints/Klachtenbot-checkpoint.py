import streamlit as st
import requests
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Laad het Nederlandse toxicity model
tokenizer = AutoTokenizer.from_pretrained("ml6team/robbert-dutch-base-toxic-comments")
model = AutoModelForSequenceClassification.from_pretrained("ml6team/robbert-dutch-base-toxic-comments")

def analyze_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    toxicity_score = probabilities[0][1].item()  # Probability of toxic class
    print(f"Toxicity: {toxicity_score}")
    return toxicity_score
    
def calculate_priority_score(toxicity_score, category, found_keywords, text):
    base_score = 0
    
    # Adjust score based on toxicity
    if toxicity_score > (tox_threshold*0.5):
        toxicity_factor = toxicity_score * 5  # Max 3 points for high toxicity
        base_score += toxicity_factor

    # Adjust score based on category
    if category in high_priority_categories:
        base_score += 3
    
    # Adjust score based on urgent keywords in the full text
    urgent_keywords = ["gevaarlijk", "onveilig", "spoed", "levensgevaar", "gewond", "overstroming", "brand", "explosie", "instorting", "giftig", "gaslek", "stroomuitval", "ongeluk", "aanrijding", "verwonding", "verstikking", "verdrinking", "bedreiging", "inbraak", "overval", "vandalisme", "agressie", "geweld", "paniek", "evacuatie", "noodsituatie", "ramp", "crisis", "epidemie", "besmetting", "vergiftiging", "ontploffing", "verzakking", "botsing", "calamiteit"]
    if any(keyword in text.lower() for keyword in urgent_keywords):
        base_score += 3

    final_score = max(1, min(round(base_score), 10))

    # Ensure the score is between 1 and 10
    return final_score

###############################################

# Definieer categorieën en trefwoorden
categories = {
    "Infrastructuur": [
        "wegen", "straat", "verkeer", "infrastructuur", "borden", "tunnel", "brug", "wegdek", "wegmarkering", 
        "rotonde", "snelweg", "stoeprand", "verkeerslichten", "fietspad", "tegel", "putdeksel", "afsluiting", 
        "fout wegdek", "slecht wegdek", "put", "stoep", "gaten in weg", "werkzaamheden", "wegenis", "wegomlegging", 
        "verkeersregelaar", "belasting", "verkeershinder", "rijstrook", "oever", "bruggen", "kruispunt", 
        "beton", "storing", "wegonderhoud", "wegconstructie", "rijbaan", "verkeersbord", "afslag", "politieblokkade", 
        "borden", "drempel", "sloot", "wegversmalling", "wegbeveiliging", "achterstallig onderhoud", "slipgevaar", 
        "fietsers", "auto", "afzetting", "straatmeubilair", "hoeken", "achteruitrijden", "snelheidsmeter", "geluidswal"
    ],
    
    "Afvalbeheer": [
        "afval", "vuilnis", "zwerfvuil", "container", "recycling", "scheiding", "plastic", "papier", "glas", 
        "groenafval", "restafval", "tuinafval", "milieu", "ophaaldienst", "afvalzak", "afvalbak", "afvalbakken", 
        "containerpark", "geur", "vuil", "verwerking", "afvalscheiding", "afvalophaal", "bak", "afvalcontainer", 
        "hondenscheet", "zwerfvuilplaag", "struiken", "plastic zak", "luiers", "borden", "onvoldoende vuilnisbakken", 
        "afvalindustrie", "restafval", "goederen", "grofvuil", "afvalinslag", "storten", "ongewenst vuil", "afvalverwerking", 
        "plastic in zee", "afvalput", "geurhinder", "vuilniswagen", "kliko", "dierafval", "afvalmonsters", 
        "afvaldistributie", "geurvervuiling", "hondenpoep", "afvalbeheer", "chemisch afval", "teveel afval", 
        "verpakking", "afvalreductie", "gemeente vuil", "gesloten vuilnisbak", "groenafvalophaling", "niet gehaalde zakken"
    ],

    "Verlichting": [
        "verlichting", "lampen", "straatverlichting", "verlichtingstekort", "lampen defect", "verlichtingspaal", 
        "licht", "verlichtingsinstallatie", "lantaarnpaal", "verlichtingsproblemen", "lamp kapot", "flikkering", 
        "onderhoud verlichting", "lampen branden niet", "energiebesparing verlichting", "lichtsterkte", "oplichtende paal", 
        "verlichtingsoplossing", "kapotte straatlamp", "fout verlichting", "verkeerde verlichting", "verlichting vervangen", 
        "verlichting uit", "lichteffecten", "brandende lamp", "lamp vervangen", "verlichtingstekort", "verlichte straat", 
        "donkere straten", "overlast verlichting", "straatverlichting dimmen", "overlast van verlichting", 
        "energiezuinige verlichting", "verlichtingsplan", "lichten", "straatlampen", "palen", "lampen schijnen", 
        "lichtvervuiling", "lantaarns", "dimbaar", "lampen storen", "knipperende verlichting", "verlichtingsinfrastructuur", 
        "led verlichting", "donkere zones", "nachtverlichting", "verlichting hangende kabels", "verlichting bij overgangen"
    ],

    "Overlast": [
        "overlast", "lawaai", "hinder", "storing", "geluid", "overlastgevers", "buurtlawaai", "lawaai overlast", 
        "storingen", "luidruchtig", "herrie", "geluidsoverlast", "overlast van verkeer", "geluidsoverlast woningen", 
        "huisdieren", "gesprek", "drummen", "hoorn", "restaurantlawaai", "luide muziek", "scooters", "motorrijders", "vuurwerk", 
        "geluidsbarrières", "overlast van apparaten", "verkeersgeluiden", "gedoe", "geluidshinder", "horecagelegenheden", 
        "restauranthinder", "misbruik", "drukte", "onwenselijk gedrag", "groepjes", "stoornis", "woningen", 
        "onrustige buurt", "buurtprobleem", "geluidsoverlast speeltuinen", "burenlawaai", "luidruchtige feestjes", 
        "geen privacy", "lawaai van voertuigen", "overlast van bewoners", "lawaai tijdens nacht", "open ramen", 
        "drukte op straat", "ergernis", "overtreding van stilte", "storing wifi", "smog", "overlast van roken", 
        "lawaai van kinderen", "probleem met afval", "afvaloverlast", "jongerenlawaai", "weergalmende geluiden"
    ],

    "Groenbeheer": [
        "groen", "tuinen", "plantsoen", "bomen", "gras", "groenvoorziening", "groenonderhoud", "tuinbeheer", 
        "planten", "bloemen", "tuin", "wilde planten", "planten in de openbare ruimte", "snoeien", "struiken", "takken", 
        "schoonmaken", "groene plekken", "plantsoenonderhoud", "groene energie", "bomenkap", "dode bomen", "bloemplantsoen", 
        "bladeren", "verwaarlozing", "onvoldoende groen", "tuinieren", "hagen", "waterbeheer", "wateroverlast", "bloei", 
        "bomen planten", "tuinservice", "plantgoed", "boomverzorging", "groencompensatie", "zaaien", "gronddoelen", 
        "ecologisch beheer", "buitenruimte", "bloemenperk", "moestuin", "wildgroei", "plantenverzorging", "kappen", 
        "bloemenperken", "terrasbeplanting", "tuinonderhoud", "groenvoorzieningen", "onderhoud bomen", "aarden", 
        "groenisolatie", "wilde bloemen", "afgevallen bladeren", "plantsoendiensten", "groenvoorzieningbeheer", "groenplan", 
        "natuurbehoud", "natuurbescherming", "groenbeheer", "milieuplan", "rondstruinen", "groenplan"
    ],

    "Waterbeheer": [
        "overstroming", "wateroverlast", "waterschade", "riolering", "watervoorziening", "afvoer", "dijk", "vloed", 
        "stormwater", "waterput", "waterkwaliteit", "afwateren", "kanaal", "rivier", "waterbeheer", "pompstation", 
        "waterpompen", "overbelaste riolering", "watervloed", "drainage", "regenwater", "waterkracht", "irrigatie", 
        "sloot", "onderwaterdorp", "droogte", "waterproblematiek", "waterafvoer", "watermolen", "waterafvoersysteem", 
        "dijkverhoging", "regensensor", "moeras", "polder", "waterberging", "vijver", "regenwatertank", "waterprijs", 
        "waterinfrastructuur", "binnendijk", "beek", "waterzuivering", "afvoerleidingen", "vloedwal", "wateropvang", 
        "ondergrondse waterpomp", "dijken", "waterproblemen", "grondwater", "schade door water", "boezem", "regenpijp", 
        "waterputting", "waterpeil", "vijverbeheer", "overstromingsgebieden", "waterpompstations", "waterschapslasten"
    ],

    "Verkeer en Mobiliteit": [
        "verkeer", "files", "verkeersdrukte", "verkeershinder", "stoplichten", "verkeersongeluk", "toegangspaden", 
        "verkeerscirculatie", "parkeren", "parkeerproblemen", "verkeersregelaar", "omleiding", "auto's", "verkeersbord", 
        "fietsers", "scooters", "motoren", "taxi's", "ov$", "trein", "bussen", "verkeersignalen", "verkeersintensiteit", 
        "snelheidsmetingen", "auto parkeren", "blokken", "snelheid", "rijstroken", "stadsverkeer", "verkeersafsluitingen", 
        "politiecontrole", "parkeerbelasting", "verkeersrondjes", "verkeersomleiding", "rijbanen", "rondrijden", "filedruk", 
        "achterstallig onderhoud", "mobility as a service", "overvolle bussen", "overtredingen", "stadsvervoer"
    ],

    "Belastingen en geldzaken": [
        "belasting", "aangifte", "inkomstenbelasting", "btw", "ozb", "gemeentebelasting", "heffing", "toeslag", "subsidie", "boete",
        "betalingsregeling", "schuld", "kwijtschelding", "bezwaar", "belastingaanslag", "belastingdienst", "toeslagen", "hypotheek",
        "lening", "sparen", "begroting", "inkomen", "uitgaven", "financiën", "rekening", "bankzaken", "verzekering", "pensioen",
        "uitkering", "bijstand", "ww", "aow", "kinderbijslag", "studiefinanciering", "zorgtoeslag", "huurtoeslag", "kinderopvangtoeslag",
        "belastingteruggave", "belastingaftrek", "vermogensbelasting", "erfbelasting", "schenkbelasting", "autobelasting", "wegenbelasting",
        "afvalstoffenheffing", "rioolheffing", "waterschapsbelasting", "precariobelasting", "toeristenbelasting", "hondenbelasting",
        "parkeergeld", "leges", "naheffing", "betalingsachterstand", "incasso", "deurwaarder", "bewindvoering", "budgetbeheer",
        "schuldhulpverlening", "faillissement", "wsnp", "belastingvrije voet", "box 3", "voorlopige aanslag", "definitieve aanslag"
]
}

###############################################

# Verkrijg de locatie via ipinfo.io
def get_location():
    try:
        response = requests.get('https://ipinfo.io')
        data = response.json()
        location = data.get('city', 'Onbekend')  # Stad, als beschikbaar
        return location
    except requests.exceptions.RequestException:
        return "Locatie onbekend"


# Functie: Analyseer klacht
def analyze_complaint(text, updated_categories, tox_threshold=-0.4):
    
    toxicity_score = analyze_toxicity(text)
    
    # Match keywords using the updated categories
    category_matches = {
        category: [word for word in keywords if word in text.lower()]
        for category, keywords in updated_categories.items()
    }
    
    # Find the category with the most matches
    matched_category = max(category_matches, key=lambda category: len(category_matches[category]))
    
    extra_urgent = toxicity_score > tox_threshold
    relevant_keywords = [word for word in category_matches[matched_category]]
    
    # Calculate priority score based on toxicity and matched keywords
    priority_score = calculate_priority_score(toxicity_score, matched_category, relevant_keywords, text)
    
    if relevant_keywords:
        return (
            f"Klacht over {matched_category}: {', '.join(relevant_keywords)}",
            matched_category,
            extra_urgent,
            relevant_keywords,
            toxicity_score,
            priority_score,
        )
    
    return (
        "Algemene klacht ontvangen",
        "Onbekend",
        extra_urgent,
        [],
        toxicity_score,
        priority_score,
    )

###############################################

if 'results' not in st.session_state:
    st.session_state.results = []

st.set_page_config(
    page_title="Klachtenbot 1.0",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed")

# Streamlit UI
st.title("🤖 Utrecht Klachtenbot 1.0 🤖")

st.write("Welkom bij **Utrecht Klachtenbot 1.0**. Ik zal ervoor zorgen dat uw klacht zo snel mogelijk afgehandeld wordt.")
user_input = st.text_area("Voer hier uw klacht voor de gemeente in. Geef alstublieft aan wat uw klacht is en - indien van toepassing - op welke locatie het probleem zich voordoet.", placeholder="Typ hier uw klacht...", height=250)

# Verkrijg de locatie
user_location = get_location()

# Parameters aanpassen
st.sidebar.header("Parameters")
tox_threshold = st.sidebar.slider("Drempel toxiciteitsscore:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# Hoge prioriteit categorieën
st.sidebar.header("Hoge prioriteit categorieën")
high_priority_categories = st.sidebar.multiselect(
    "Kies categorieën:",
    options=list(categories.keys()),
    default=["Infrastructuur", "Verkeer en Mobiliteit", "Waterbeheer"]
)

# Sidebar met categorieën en hun trefwoorden
st.sidebar.header("Trefwoorden per categorie")
# Geef de mogelijkheid om de trefwoorden per categorie aan te passen
updated_categories = {}
for category, words in categories.items():
    with st.sidebar.expander(f"{category}"):
        updated_keywords = st.text_area(f"Trefwoorden:", value=", ".join(words), height=100)
        updated_categories[category] = [word.strip() for word in updated_keywords.split(",")]

# Output
if user_input:
    advice, category, extra_urgent, found_keywords, toxicity_score, priority_score  = analyze_complaint(user_input, updated_categories, tox_threshold)

    # Append result to the list
    st.session_state.results.append({
        "klacht": user_input,
        "plaats": user_location,
        "categorie": category,
        "prioriteitsscore": priority_score
    })

    # Show results
    st.header("Analyse")
    st.write(f"**Prioriteitsscore:** {priority_score}/10")
    if extra_urgent:
        st.write("⚠️ Deze klacht is aangemerkt als zeer dringend. ⚠️")
    st.write(f"**Categorie:** {category}")
    if found_keywords:
        st.write(f"**Herkende trefwoorden:** {', '.join(found_keywords)}")
    st.write(f"**Locatie van de indiener:** {user_location}\n\n")


else:
    st.write("Druk op Ctrl+Enter om de klacht te analyseren.")


# Display results table
if st.session_state.results:
    st.header("Overzicht van klachten")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df, use_container_width=False, hide_index=True)