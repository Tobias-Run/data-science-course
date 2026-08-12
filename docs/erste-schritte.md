# Erste Schritte — vom leeren Rechner zum ersten Lauf

Diese Anleitung setzt nichts voraus. Jeder Schritt hat eine Kontrolle: wenn du
nicht siehst, was dort steht, geh nicht weiter — spring stattdessen ans Ende zu
„Wenn etwas schiefgeht".

Zeitbedarf beim ersten Mal: etwa 30 Minuten, das meiste davon Downloads.

---

## Schritt 1 — Projekt herunterladen

Öffne ein Terminal (Windows: **PowerShell**, macOS: **Terminal**) und gib ein:

```bash
git clone https://github.com/Tobias-Run/data-science-course.git
cd data-science-course
git checkout claude/worldclaw-3d-generator-wbao6q
```

**Kontrolle:**

```bash
git branch --show-current
```

→ muss `claude/worldclaw-3d-generator-wbao6q` ausgeben.

---

## Schritt 2 — Python 3.11 prüfen

Blender lässt sich als Python-Baustein nur mit **genau Version 3.11** benutzen.
Nicht 3.12, nicht 3.13.

**Windows:**
```powershell
py -3.11 --version
```

**macOS / Linux:**
```bash
python3.11 --version
```

**Kontrolle:** → `Python 3.11.x`

**Fehlt sie?** Von [python.org/downloads](https://www.python.org/downloads/)
eine 3.11er Version holen. Beim Windows-Installer unbedingt
**„Add python.exe to PATH"** anhaken. Danach Terminal schließen, neu öffnen,
Prüfung wiederholen.

---

## Schritt 3 — Abgetrennte Arbeitsumgebung anlegen

Damit die Installation dein System nicht durcheinanderbringt.

**Windows:**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Kontrolle:** Vor deinem Eingabeprompt steht jetzt `(.venv)`.

> Diesen Aktivierungsbefehl brauchst du **jedes Mal neu**, wenn du ein frisches
> Terminal öffnest. Ohne ihn findet der Rechner die Befehle nicht.

---

## Schritt 4 — Installieren

```bash
pip install -e ".[blender,dev]"
```

Das lädt unter anderem Blender herunter, rund **1 GB**. Es darf ein paar
Minuten dauern.

**Kontrolle:**

```bash
worldclaw --help
```

→ zeigt eine Liste von Befehlen: `layout, terrain, blender, plan, doctor, all, …`

---

## Schritt 5 — LM Studio einrichten

1. **Modell laden.** In LM Studio auf **Discover** (Lupe) gehen und ein
   Instruct-Modell herunterladen. Als Anhaltspunkt: etwa 7–14 Milliarden
   Parameter, Quantisierung `Q4_K_M`. Achte auf Modelle, deren Beschreibung
   *structured output*, *function calling* oder *tool use* erwähnt — das ist
   genau die Fähigkeit, die hier gebraucht wird.
2. **Modell aktivieren.** Oben in der Modellauswahl das geladene Modell wählen.
3. **Server starten.** Linke Seitenleiste → **Developer** (Symbol `>_`) →
   Schalter auf **Running**.

**Kontrolle:** In LM Studio steht die Adresse, üblicherweise
`http://localhost:1234`. Im Terminal:

```bash
worldclaw llm-check
```

→ listet dein Modell auf. **Notiere dir den genauen Namen** — den brauchst du
gleich.

---

## Schritt 6 — Startprüfung

```bash
worldclaw doctor --llm-model DEIN-MODELLNAME
```

Das prüft alles auf einmal, inklusive der wichtigsten Frage: ob dein Modell
sich wirklich an ein vorgegebenes Antwortformat hält.

**Kontrolle:** Ganz unten muss stehen: **`ready for a full run`**

Warnungen (`warn`) sind in Ordnung — die betreffen optionale Teile. Nur `FAIL`
hält dich auf, und jede Fehlermeldung sagt dir, was zu tun ist.

---

## Schritt 7 — Der Lauf

```bash
worldclaw all --prompt "Eine ausgetrocknete Schlucht in rotem Sandstein, Geröll am Grund" --llm-model DEIN-MODELLNAME --run mein-erster-lauf --render --save-blend
```

Vier Abschnitte laufen nacheinander durch:

| Abschnitt | Dauer | Was passiert |
|---|---|---|
| `1/4 planning` | Sekunden | Dein Satz wird zu einer Geländebeschreibung |
| `2/4 terrain` | ~20 s | Höhenfeld, Materialien, Bewuchs |
| `3/4 blender` | 1–3 min | Mesh und drei Bilder |
| `4/4 checks` | Sekunden | Automatische Qualitätsprüfung |

**Kontrolle:** Am Ende steht `10 passed, 0 failed` und ein Pfad zu einem Bild.

---

## Schritt 8 — Ergebnis ansehen

Alles liegt in `artifacts/mein-erster-lauf/`. In dieser Reihenfolge anschauen:

1. **`blender/render_cam_ground.png`** — die Landschaft auf Augenhöhe (1,70 m),
   stehend an der steilsten Kante. Das ist der eigentliche Test.
2. **`terrain/preview_hillshade.png`** — das Höhenfeld von oben.
3. **`plan/layout.png`** — die Regionskarte, aus der alles entstand.
4. **`plan/intent.json`** — was dein Modell aus dem Satz herausgelesen hat.
   Alles, was du *nicht* gesagt hast, muss dort `null` sein.
5. **`blender/terrain.blend`** — in Blender öffnen und selbst herumlaufen.

---

## Wenn etwas schiefgeht

**`worldclaw: command not found` / `wird nicht erkannt`**
Die Umgebung ist nicht aktiv. Schritt 3 wiederholen. Notfalls funktioniert
statt `worldclaw ...` auch `python -m worldclaw.cli ...`.

**`doctor` meldet `llm endpoint unreachable`**
Der Server in LM Studio läuft nicht. Developer-Tab, Schalter auf **Running**.
Läuft er auf einem anderen Port, ergänze `--llm-url http://localhost:PORT/v1`.

**`doctor` meldet `structured output FAIL`**
Dein Modell hält sich nicht an das Antwortformat. LM Studio aktualisieren, oder
ein größeres Instruct-Modell nehmen. Sehr kleine Modelle scheitern hier
regelmäßig.

**`bpy warn: not importable`**
Falsche Python-Version. Gelände wird trotzdem gebaut, aber es gibt keine
Bilder. Schritt 2 und 3 mit 3.11 wiederholen.

**Der Lauf bricht mittendrin ab**
Nicht schlimm: jede Stufe ist zwischengespeichert. Denselben Befehl einfach
noch einmal ausführen — er setzt an der abgebrochenen Stelle wieder an.

**Blender dauert ewig**
Läuft absichtlich auf der CPU, damit die Grafikkarte frei bleibt. Mit
`--samples 12` wird es schneller und körniger, mit `--decimate 2` deutlich
schneller und gröber.

**Der Plan ergibt keinen Sinn**
Zuerst `plan/intent.json` ansehen. Steht dort ein Biom, das dein Satz nie
erwähnt hat, ignoriert dein Modell die Anweisung — dann hilft ein größeres.
Ist die Extraktion sauber und nur der Plan seltsam, ist das eine
Entwurfsentscheidung des Planers; was er ergänzt hat, listet
`scene_plan.json` unter `inferred_fields` auf.

---

## Und dann?

Andere Prompts ausprobieren, gern gegensätzliche — etwas Flaches („ein weites
Dünenmeer") und etwas Schroffes („zerklüftete Felsnadeln über einem Talkessel").
Am Hang zeigen sich Fehler, in der Ebene nie.
