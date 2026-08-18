# Lizenzprüfung: Microsoft Rebrand Registry als Vorbild

**Geprüftes Vorbild:** `loryanstrant/Microsoft-Rebrand-Registry`
(https://github.com/loryanstrant/Microsoft-Rebrand-Registry, Live: https://www.msrebrandregistry.com)
**Prüfstand:** Klon von `main`, HEAD `f962112` („Merge pull request 'Refine former-name header across pages' (#11)", 18.08.2026), vollständige Historie (53 Commits, erster Commit `68e521a` vom 14.08.2026)
**Prüfdatum:** 18.08.2026

---

## Ergebnis in einem Satz

**Keine Lizenz vorhanden — weder im Repository, noch in der Historie, noch auf der Seite.** Damit gilt „all rights reserved": Fork und Code-Übernahme sind nicht gedeckt. Nach dem Entscheidungsbaum des Briefings ist Fall 3 einschlägig: **Neuimplementierung**, keine Code-Übernahme.

---

## 1. Lizenzdateien im Repository-Root und in `.github/`

| Geprüfter Pfad | Befund |
| --- | --- |
| `LICENSE`, `LICENSE.md`, `LICENSE.txt` | nicht vorhanden |
| `COPYING`, `COPYING.md` | nicht vorhanden |
| `NOTICE`, `NOTICE.md` | nicht vorhanden |
| `COPYRIGHT` | nicht vorhanden |
| `.github/` (gesamtes Verzeichnis) | **existiert nicht** — auch keine `.github/LICENSE`, kein `FUNDING.yml`, keine Templates |

Methode: `find . -iname '*licen[cs]e*' -o -iname 'COPYING*' -o -iname 'NOTICE*' -o -iname 'COPYRIGHT*'` über den gesamten Arbeitsbaum (ohne `.git`) — **null Treffer**.

Der vollständige Dateibestand des Repositories (ohne die 60 Logodateien unter `src/assets/logos/`) umfasst genau: `.gitignore`, `README.md`, `TECHNICAL.md`, `analysis.html`, `index.html`, `azure-static-web-apps.yml.example`, `package.json`, `staticwebapp.config.json`, `scripts/{package-deployment,validate-data}.js`, `src/{analysis,app,dates,former-site-names}.js`, `src/styles.css`, `src/data/products.json`, `src/assets/site-mark.svg`, `tests/{analysis,data}.test.js`. Eine Lizenzdatei ist darin nicht enthalten.

## 2. Feld `license` in `package.json`

Datei vollständig geprüft. Vorhandene Schlüssel: `name`, `version`, `private: true`, `type`, `scripts`, `engines`.

- **`license`: nicht gesetzt.**
- `"private": true` verhindert lediglich ein versehentliches npm-Publish; es ist **keine** Lizenzaussage und erst recht keine Rechteeinräumung.
- Ein fehlendes `license`-Feld bei `private: true` ist npm-seitig zulässig — es sagt über die Rechtelage nichts aus, außer dass keine erteilt wurde.

## 3. Lizenzabschnitt in `README.md` und `TECHNICAL.md`

Beide Dateien vollständig gelesen und zusätzlich per Regex durchsucht
(`grep -niE 'licen[cs]e|copyright|\(c\)|©|all rights reserved|MIT|Apache|GPL|BSD|ISC'`).

- `README.md`: **kein Lizenzabschnitt.** Gliederung ist „Data methodology", „Analysis", „Contribute", „Technical documentation". Der Abschnitt „Contribute" lädt zu Beiträgen über das öffentliche Repository ein, **ohne** Lizenz-, CLA- oder Rechteeinräumungshinweis. Die beiden Regex-Treffer in `README.md` (Zeilen 17, 27) enthalten das Wort „includes"/„labelled" im Fließtext und betreffen den Datenumfang, nicht die Lizenz.
- `TECHNICAL.md`: **kein Lizenzabschnitt.** Der einzige Regex-Treffer (Zeile 41) betrifft „token-based manual releases" beim Azure-Deployment.
- Ebenfalls ohne Lizenzangabe geprüft: `.gitignore`, `staticwebapp.config.json`, `azure-static-web-apps.yml.example`.

## 4. Header-Kommentare in den Quelldateien

Erste 12 Zeilen jeder Quelldatei einzeln gesichtet, zusätzlich Volltextsuche über `src/`, `scripts/`, `tests/`, `index.html`, `analysis.html` nach `licen[cs]e|copyright|©|(c) 20|rights reserved`: **null Treffer**.

| Datei | Beginn | Lizenzheader |
| --- | --- | --- |
| `src/app.js` | direkt mit `import { … } from './dates.js';` | nein |
| `src/analysis.js` | direkt mit `import { … } from './dates.js';` | nein |
| `src/dates.js` | direkt mit `export function parseDate(…)` | nein |
| `src/former-site-names.js` | direkt mit `export const FORMER_SITE_NAMES = [` | nein |
| `src/styles.css` | direkt mit `:root{--ink:#172033;…}` (minifiziert, einzeilig) | nein |
| `scripts/validate-data.js` | direkt mit `import { readFile } …` | nein |
| `scripts/package-deployment.js` | direkt mit `import { cp, mkdir, … } …` | nein |
| `tests/*.test.js` | ohne Header | nein |
| `index.html`, `analysis.html` | `<!doctype html>` … `<head>` ohne Lizenz-Meta | nein |

Keine SPDX-Kennung (`SPDX-License-Identifier`) in irgendeiner Datei.

## 5. Footer und Impressum der Live-Seite

Die Live-Hosts sind aus dieser Umgebung **nicht erreichbar** — sowohl `https://www.msrebrandregistry.com` als auch der in `TECHNICAL.md` genannte Azure-Host `https://wonderful-ocean-034ff8f1e.7.azurestaticapps.net` werden vom Egress-Proxy blockiert (`EGRESS_BLOCKED`). Ersatzweise wurde der **ausgelieferte Quelltext** der beiden Seiten auf `main` geprüft; das Repository ist laut `TECHNICAL.md` unverändert das Deployment-Artefakt (App location `/`, kein Build-Schritt).

Footer `index.html` (Zeilen 107–110), wörtlich:

> Independent research project. Microsoft product names, logos and trademarks belong to Microsoft. No rebrands were harmed in the making of this registry.
> Contribute on GitHub · \<as-of-Datum\>

Footer `analysis.html`, wörtlich identisch bis auf „…in the making of this forecast."

Bewertung: Der Footer regelt **ausschließlich Markenrechte Dritter (Microsoft)**. Er enthält **keine** Lizenz für den Code der Seite, keinen Copyright-Vermerk des Autors, kein Impressum und keinen Link auf Nutzungsbedingungen. Ein Impressum existiert auf keiner der beiden Seiten; weitere Seiten gibt es nicht (`staticwebapp.config.json` hat nur einen Navigation-Fallback).

*Offener Restpunkt:* Der Footer der öffentlich erreichbaren Domain konnte nicht mit eigenen Augen geprüft werden. Weicht er vom Repository-Stand ab, wäre das nachzuholen; angesichts des identischen Deployment-Pfads ist eine Abweichung unwahrscheinlich.

## 6. Git-Historie: nachträglich entfernt oder hinzugefügt?

- `git log --all --diff-filter=ADMR --name-only` über alle Branches, gefiltert auf `licen[cs]e|copying|notice|copyright`: **null Treffer**. Eine Lizenzdatei wurde also **nie** angelegt und folglich auch nie entfernt.
- Vollständige Liste aller je committeten Pfade (`git log --all --name-only`) mit dem aktuellen Bestand abgeglichen: keine gelöschten Dateien außer Logo-Assets.
- Geprüfte Refs: `origin/main`, `origin/feat/table-timeline-poc`. Keine Tags.
- Auch `.github/` taucht in der gesamten Historie nie auf.

Das Fehlen der Lizenz ist damit kein Versehen einer gekürzten Dateiliste in der GitHub-Oberfläche, sondern der tatsächliche Zustand seit dem ersten Commit.

**Zusatzbeleg:** Die GitHub-Projektseite zeigt in der „About"-Sidebar keinen License-Eintrag; die Sidebar enthält nur die Beschreibung „The source for a site that tracks Microsoft cloud product renames." GitHubs Lizenzerkennung findet also ebenfalls nichts. (Die REST-API `api.github.com` ist aus dieser Umgebung nicht abrufbar — HTTP 403 vom Proxy.)

## 7. Lizenzen der eingebundenen Abhängigkeiten

Das Projekt hat **keine** Abhängigkeiten: In `package.json` fehlen `dependencies`, `devDependencies` und `peerDependencies` vollständig. Es existiert **kein** Lockfile (`package-lock.json`, `yarn.lock`, `pnpm-lock.yaml` — alle nicht vorhanden) und kein `node_modules`.

Berührte Fremdsoftware nur zur Laufzeit der npm-Skripte:

| Paket | Verwendung | Lizenz | Relevanz für uns |
| --- | --- | --- | --- |
| `serve@14.2.4` | `npm start` ruft `npx --yes serve@14.2.4 .` — reiner lokaler Entwicklungsserver | **MIT** (laut npm-Registry-Metadaten zu Version 14.2.4) | gilt unabhängig weiter; MIT erlaubt uns dieselbe Nutzung ohne Weiteres |
| Node.js `node --test`, `node:fs/promises` | Tests und Skripte | Node-eigene Lizenz (MIT-basiert) | unkritisch, Standardbibliothek |

Kein CDN, kein externes Skript, kein externes Stylesheet: Die einzigen HTTP-Verweise in HTML/JS/CSS sind redaktionelle Links (mscloudlogos.com, letmecorrectthatforyou.com, m365maps.com, rebrandedbyms.com, das GitHub-Repository). Diese Abhängigkeitslizenzen decken **nur** die jeweiligen Pakete ab — sie sagen nichts über den Code des Vorbilds aus.

## 8. Schriften, Icons und `site-mark.svg`

| Asset | Herkunft | Lizenz | Folge für uns |
| --- | --- | --- | --- |
| Schriften | keine Fontdateien im Repository. CSS nutzt ausschließlich Systemschriften: `font-family:"Segoe UI",system-ui,-apple-system,sans-serif` und `font-family:Georgia,serif` | keine Lizenzfrage (keine Auslieferung von Fontdateien); „Segoe UI" ist eine Microsoft-Schrift, die nur benutzt wird, wenn sie lokal installiert ist | wir liefern ebenfalls keine Fonts aus, wählen aber eine eigene Systemschrift-Kaskade ohne „Segoe UI" |
| Icon-Bibliothek | keine (Font Awesome o. ä. nicht eingebunden); Symbole sind Unicode-Zeichen im Markup | — | unkritisch |
| `src/assets/site-mark.svg` | handgezeichnetes SVG des Autors (Zifferblatt mit Fragezeichen), 14 Zeilen, ohne Lizenzangabe | **keine Lizenz → all rights reserved** | **nicht übernehmen**, auch nicht abgewandelt. Wir brauchen eine eigene Wortmarke/Signet |
| `src/assets/logos/*` (60 Dateien, SVG/PNG) | Microsoft-Produktlogos, laut `README.md` aus der Sammlung „Microsoft Cloud Logos" (mscloudlogos.com) | Microsoft-Markenrecht, keine offene Lizenz | für uns ohnehin ausgeschlossen — Abschnitt 9 des Briefings verbietet SAP-Logos vollständig |

---

## Bewertung und Konsequenz

Nach § 2 des Briefings greift der dritte Zweig des Entscheidungsbaums: **keine Lizenz gefunden, Lage eindeutig, nicht bloß unklar.** Ohne Rechteeinräumung gilt das gesetzliche Urheberrecht — die Vervielfältigung, Bearbeitung und Veröffentlichung des Codes ist untersagt. Das schließt den GitHub-Fork ein: Der Fork-Button erteilt keine Lizenz, sondern setzt eine voraus (GitHubs ToS decken nur das Forken *innerhalb* von GitHub, nicht Weiterverwendung und eigenständige Veröffentlichung).

**Nicht übernehmbar:**

- jeglicher JavaScript-Code aus `src/` und `scripts/` — auch nicht einzelne Funktionen wie `median()`, `parseDate()` oder `monthDiff()` in ihrer konkreten Ausformulierung
- `src/styles.css` vollständig, inklusive Farbpalette und Layoutwerten in ihrer konkreten Zusammenstellung
- HTML-Struktur und Klassennamen von `index.html` / `analysis.html` als Ganzes
- sämtliche Texte: Überschriften, Erläuterungen, Methodenhinweise, Footer-Formulierungen, die Liste in `former-site-names.js`
- `site-mark.svg`
- `products.json` als Datenbank (Datenbankherstellerrecht, § 87a UrhG, unabhängig vom Urheberrecht an den Einzelangaben)

**Frei verwendbar, weil nicht schutzfähig:**

- die Idee einer belegpflichtigen Rebrand-Registry mit Analyseseite
- die Informationsarchitektur: zwei Seiten, Tabelle mit Laufzeitbalken, alphabetische Sprungmarken, Filter
- das Datenmodell dem Konzept nach: Produkt mit Perioden, Start/Ende, Qualifier, Quellenreferenzen — Feldnamen und Schema formulieren wir selbst aus
- die Kennzahlen und ihre Definition: Median der Namenslaufzeiten, Familienfrequenz, Anzahl früherer Identitäten, gewichteter Risk Index
- die Gewichtungsanteile (45 / 35 / 20) als reine Zahlen
- die recherchierten historischen Fakten selbst (Daten sind keine Werke); wir erheben sie ohnehin für SAP neu und aus eigenen Quellen

**Vorgehen:** Neuimplementierung ohne Blick in den fremden Code während des Schreibens. Der Klon liegt außerhalb unseres Projektverzeichnisses unter `/workspace/loryanstrant/microsoft-rebrand-registry` und wird nicht in unser Repository übernommen. Im Footer nennen wir das Vorbild als konzeptionelle Anregung — das ist eine redaktionelle Referenz, keine lizenzrechtlich erforderliche Attribution, und darf nicht als Zusammenarbeit oder Billigung missverstanden werden.

---

## Anhang: Entwurf einer Anfrage an den Autor (GitHub-Issue)

Zu senden an `loryanstrant/Microsoft-Rebrand-Registry`, sobald freigegeben. Die Arbeit läuft unabhängig davon weiter; eine Antwort ist nicht abzuwarten.

**Titel:** Question: which licence applies to this repository?

**Text:**

> Hi Loryan,
>
> The Rebrand Registry is a genuinely nice piece of work — the insistence on a cited source for every name period is what makes it more than a list.
>
> I'm building an equivalent site for SAP products (SAP has been at least as busy on this front, and the acquisitions add a category of their own). I'd like to be clear about the boundaries before I go further.
>
> I couldn't find a licence: no `LICENSE` file, no `license` field in `package.json`, and no licence section in the README or the technical guide, at any point in the repository's history. So I'm treating the code, CSS and copy as all rights reserved, and I'm writing my site from scratch rather than forking. My project takes the general idea — an evidence-first registry plus an analysis page with a semi-serious risk index — but no code, styling or wording from yours.
>
> Two questions:
>
> 1. Is the absence of a licence deliberate, or simply not gotten around to yet? If you ever add a permissive one, I'd be glad to know.
> 2. Would you be comfortable with my site crediting yours in the footer as the conceptual inspiration, with a link? Happy to word it however you prefer, or to leave it out entirely.
>
> Either answer is fine — I'd just rather ask than assume.
>
> Thanks,
> …

---

## Nachvollziehbarkeit

Sämtliche Befunde stammen aus dem lokalen Klon von `main` unter `/workspace/loryanstrant/microsoft-rebrand-registry` (HEAD `f962112`) sowie der GitHub-Projektseite. Die beiden Live-Hosts konnten wegen der Egress-Sperre nicht abgerufen werden; das ist oben an Ort und Stelle vermerkt. Reproduktion der Kernprüfung:

```bash
git clone https://github.com/loryanstrant/microsoft-rebrand-registry
cd microsoft-rebrand-registry
find . -path ./.git -prune -o -iname '*licen[cs]e*' -print -o -iname 'COPYING*' -print -o -iname 'NOTICE*' -print
grep -rniE 'licen[cs]e|copyright|©|rights reserved|SPDX' . --exclude-dir=.git
git log --all --diff-filter=ADMR --name-only --pretty=format:'%h %s' | grep -iE 'licen[cs]e|copying|notice'
```
