# Denisa

Obrázky v této složce se převádějí na omalovánky podle čísel (A4 PDF) skriptem `omalovanky_prevod.py`:

```
pip install numpy scipy opencv-python-headless pillow reportlab
python omalovanky_prevod.py vstup.jpg vystup.pdf
```

- Zrnité fotky potřebují `--odsum 12`, jinak se motiv rozpadne na mozaiku mnohoúhelníků.
- `--min-plocha` je v mm² na papíře (výchozí 7), `--hladkost` vyhlazuje tvary (výchozí 5); větší hodnoty = větší a méně početné plochy.
- Po převodu výsledek zkontroluj vykreslením PDF (`pdftoppm`).
- Vstupní obrázky a vygenerovaná PDF se necommitují bez výslovného pokynu.
