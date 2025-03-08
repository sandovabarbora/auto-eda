# AutoEDA - Automatická Explorativní Analýza Dat

Interaktivní aplikace pro průzkumnou analýzu dat s využitím Streamlitu. Aplikace umožňuje rychlou vizualizaci a analýzu dat bez nutnosti psaní kódu.

## Funkce

- **Nahrání dat** - import z CSV, Excel, JSON, Parquet nebo API
- **Přehled dat** - základní statistiky a hodnocení kvality dat
- **Chybějící hodnoty** - detekce a analýza chybějících hodnot
- **Distribuce** - vizualizace rozložení numerických a kategorických proměnných
- **Korelace** - analýza vztahů mezi proměnnými
- **Odlehlé hodnoty** - identifikace a vizualizace outlierů

### Pokročilé funkce

- **Redukce dimenzí (PCA)** - snížení počtu dimenzí se zachováním informační hodnoty
- **Clustering** - segmentace dat do přirozených skupin
- **Statistické testy** - ověření hypotéz o datech
- **Časové řady** - analýza trendu a sezónnosti
- **Cross tabulky** - analýza vztahů mezi kategorickými proměnnými
- **KPI dashboard** - sledování klíčových ukazatelů výkonnosti
- **Kohortní analýza** - sledování chování skupin v čase

## Instalace

1. Naklonuj repozitář:
   ```bash
   git clone [URL repozitáře]
   cd autoeda
   ```

2. Vytvoř virtuální prostředí a nainstalujte závislosti:
   ```bash
   make setup
   source venv/bin/activate
   make install
   ```

3. Alternativně může2 vytvořit virtuální prostředí ručně:
   ```bash
   python -m venv venv
   
   # Na Windows
   venv\Scripts\activate
   
   # Na Linux/macOS
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

## Spuštění aplikace

Pro spuštění aplikace použij:
```bash
make run
```

Nebo přímo:
```bash
streamlit run app.py
```

Aplikace bude dostupná na adrese http://localhost:8501

## Požadavky na systém

- Python 3.8 nebo novější
- Všechny závislosti uvedené v souboru `requirements.txt`
