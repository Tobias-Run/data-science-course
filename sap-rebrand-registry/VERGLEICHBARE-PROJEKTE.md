# Vergleichbare Projekte

**Stand:** 18.08.2026
**Frage:** Gibt es bereits ein Projekt, das SAP-Produktumbenennungen so erfasst, wie wir es vorhaben?

**Antwort:** Nein. Kein Register mit Namensperioden, Datumsbelegen und daraus berechneten Kennzahlen. Die Nische ist offen.

## Erhebungsmethode und ihre Grenzen

Recherchiert wurde ausschließlich per Websuche. Der Egress-Proxy dieser Umgebung sperrt `sap.com`, `help.sap.com`, `news.sap.com`, `web.archive.org` und `btp.udina.de`; die genannten Seiten konnten also bis auf eine Ausnahme nicht selbst geöffnet werden. Der Suchindex ist US-zentriert — ein rein deutschsprachiges oder frisch gestartetes Projekt kann durchgerutscht sein. Negativbefund also mit Vorbehalt.

Ausnahme: Der UDINA Rebranding Guide lag als Textexport vor und wurde vollständig ausgewertet (siehe unten).

## Was es gibt

### 1. Bestandslisten ohne Zeitachse

- [Wikipedia, „List of SAP products"](https://en.wikipedia.org/wiki/List_of_SAP_products) — der nächste Nachbar. Umbenennungen erscheinen als Nebensatz im Fließtext („SAP XI was renamed SAP PI from release 7.0"), ohne Datum, ohne Einzelbeleg, ohne Perioden.
- [HandWiki](https://handwiki.org/wiki/Software:List_of_SAP_products), [Software Wiki bei Fandom](https://software.fandom.com/wiki/List:SAP_products) — im Wesentlichen Wikipedia-Ableger.
- Produktartikel wie [SAP BTP](https://en.wikipedia.org/wiki/SAP_BTP) und [BusinessObjects](https://en.wikipedia.org/wiki/BusinessObjects) — brauchbare Einzelchroniken, guter Rechercheeinstieg, kein Datensatz.

**Maßstab:** Wikipedia ist zugleich Einstieg und Messlatte. Wenn ein Eintrag von uns nicht mehr belegt als der entsprechende Wikipedia-Absatz, hat er keinen Grund zu existieren.

### 2. Bereichsbezogene Stichtagstabellen

[UDINA BTP, „Rebranding Guide"](https://btp.udina.de/service/sap/rebranding.html), UNIORG Cloud Services, © 2020–2026. Ausführlich ausgewertet, siehe eigener Abschnitt unten.

### 3. Erstanbieter-Hinweise, punktuell

SAP dokumentiert Umbenennungen nur lokal dort, wo Doku oder Schulungsmaterial betroffen sind:
- [„Rebrandings", SAP Help Portal (NW 7.31)](https://help.sap.com/doc/saphelp_nw73ehp1/7.31.19/en-US/3a/71895360b93d58e10000000a174cb4/content.htm)
- [„Explaining how the Recent Rebranding Displays in This Material", learning.sap.com](https://learning.sap.com/learning-journeys/developing-business-processes-with-sap-process-orchestration/explaining-how-the-recent-rebranding-displays-in-this-material_b38d8d87-cb88-4f89-abe8-c724049e8364)

Für uns Primärquellen, keine konkurrierende Aufbereitung.

### 4. Episodische Kommentare

- [HackingSAP, „SAP BI product name shake-up: the 2017 edition"](https://www.hackingsap.com/blog/sap-bi-product-name-shake-up-the-2017-edition) — schon der Titel verrät, dass jemand mehrfach angesetzt und nie ein fortlaufendes Register daraus gemacht hat.
- [„SAP product naming confusion", SAP Community, 2011](https://blogs.sap.com/2011/01/10/sap-product-naming-confusion/)
- [„SAP BusinessObjects Product Name Change", SAP Community, 2010](https://blogs.sap.com/2010/09/10/sap-businessobjects-product-name-change/)

Momentaufnahmen mit Verfallsdatum. Als Belegkandidaten der Kategorie „Blog" nach §7 nur nachrangig verwendbar.

### 5. GitHub

Kein Datensatz, kein Tracker, keine Awesome-Liste zum Thema gefunden.

### 6. Das Umfeld des Vorbilds

Das „Get Your Story Straight"-Netzwerk der Microsoft Rebrand Registry — [mscloudlogos.com](https://www.mscloudlogos.com/), [letmecorrectthatforyou.com](https://www.letmecorrectthatforyou.com/), dazu [m365maps.com/renames](https://m365maps.com/renames.htm) und rebrandedbyms.com — ist durchgehend Microsoft-only. Deckt sich mit der Annahme in §9 des Briefings: Ein SAP-Äquivalent zu mscloudlogos.com existiert nicht.

## Der UDINA Rebranding Guide im Detail

Der einzige Fund, der einer Umbenennungstabelle nahekommt. Vollständig ausgewertet.

**Was er ist:** Eine Momentaufnahme *eines* Ereignisses — die Retirement der Marke „SAP Cloud Platform" zum **18. Januar 2021**. Rund 200 Zeilen Alt→Neu in sechs Abschnitten: General, Integration and Extension Services, Third-Party Services, Environments and Runtimes, Editions, Add-Ons and Tools. Gepflegt von UNIORG Cloud Services als Nachschlagewerk für Beratungsprojekte. Er beantwortet „Wie heißt das Ding heute?", nicht „Wie oft wurde es umbenannt?".

**Abgrenzung:**

| | UDINA Rebranding Guide | Unser Register |
| --- | --- | --- |
| Granularität | Services, Editionen, SDKs, Add-Ons, Preis-SKUs | Produkte |
| Zeit | ein Stichtag, keine Vorgeschichte, keine Fortsetzung | vollständige Namensketten mit Perioden |
| Beleg | ein einziger Quellenlink in der gesamten Tabelle | mindestens eine Quelle je Periode, Datumspräzision erhalten |
| Zweck | Nachschlagen | Analyse, Median, Risk Index |
| Umfang | SAP Cloud Platform / BTP | sieben Familien |

Der entscheidende Unterschied: Bei UDINA sind die ~200 Zeilen 200 Einträge. Bei uns ist derselbe Vorgang **ein** Übergang je betroffenem Produkt (`transition: rename`, `qualifier: effective`, `start: 2021-01-18`). Würden wir Katalogzeilen zählen statt Produktidentitäten, bekäme Platform & Dev einen Median, der jede andere Familie erschlägt.

**Rechtliche Behandlung:** Fremdes Werk, © UNIORG Cloud Services, ohne offene Lizenz. Verwendung ausschließlich als Suchhilfe nach §7 des Briefings, keine Übernahme der Tabelle. Die Fakten stammen ersichtlich aus SAPs eigener Umbenennungsliste vom Januar 2021 und sind gegen diese Primärquelle zu verifizieren, sobald `sap.com` erreichbar ist.

**Konkrete Rechercheeinstiege daraus:**

- Stichtag 18.01.2021 für die Massenumbenennung SAP Cloud Platform → SAP BTP
- SAP Cloud Platform Enterprise Messaging → SAP Event Mesh, 22.02.2021, mit SAP-Blog als Quelle — ein Nachzügler zur Welle, also eigener Eintrag
- SAP Cloud Platform, SAP Data Hub service → SAP Data Intelligence service — Kandidat für Data & Analytics
- Kette Portal → Launchpad service → Work Zone → Build Work Zone: in der Seitennavigation sichtbar, in der Tabelle gar nicht vorhanden — Musterbeispiel dafür, was eine Stichtagstabelle strukturell nicht zeigen kann

**Zwei Fallen:**

1. Der Guide führt „SAP Cloud Platform Build → SAP Build" (2021). Das ist *nicht* das heutige SAP Build (2022, aus AppGyver hervorgegangen). Zwei verschiedene Produkte mit demselben Namen, elf Jahre auseinander. Zu verifizieren; falls bestätigt, ein Testfall für die Regel „`id` ändert sich nie".
2. Zeilen mit dem Kommentar „no longer available as a separate service; it is a capability within SAP Integration Suite" sind Einstellungen, keine Umbenennungen — nach §9 außerhalb unseres Umfangs. Dasselbe gilt für alle mit „Retired." markierten Zeilen.

## Folgerung für die Abgrenzung

Die Unterscheidung liegt nicht im Thema, sondern in drei Dingen, die keine der gefundenen Quellen leistet:

1. Datumspräzision mit Beleg je Periode
2. die Typisierung `rename` / `assimilation` / `generation`
3. berechnete statt behauptete Kennzahlen

## Offener Punkt

Der UDINA-Guide legt ein Muster offen, das das Datenmodell aus §3 nicht abbildet: SAP benennt nicht einzeln um, sondern in Schüben. Ein optionales `wave`-Feld auf Periodenebene (freier Slug, z. B. `btp-2021`) würde auf der Analyseseite einen Abschnitt tragen, den das Vorbild nicht haben kann. Entscheidung steht aus.
