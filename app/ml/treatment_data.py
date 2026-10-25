"""What to do about each disease - curated, bilingual, and deliberately cautious.

Rules this file follows (a defense panel will check them):
- Active ingredients only, never brand doses. In the Philippines, use only
  FPA-registered products (Fertilizer and Pesticide Authority) at the dose,
  interval and pre-harvest interval on THEIR label.
- No cure is claimed for viruses; the advice is removal and vector control.
- Every entry lists references to verify against. They are general extension
  sources (university IPM guides, CIMMYT), not a sign-off by the DA.
- "escalate" says when to bring a sample to the Municipal Agriculture Office.

The local adviser (app/ml/adviser.py) answers only from this text, so what is
here is also the limit of what it will tell a farmer.
"""

LABEL_RULE = {
    "en": "Use only FPA-registered products. Follow the label for dose, spray interval, "
          "protective gear and the days to wait before harvest.",
    "tl": "FPA-registered na products lang ang gamitin. Sundin yung label: gaano karami, gaano "
          "kadalas mag-spray, anong protective gear ang isusuot, at ilang araw ang hihintayin bago "
          "mag-harvest.",
}

# Asked as "can I still sell / eat / harvest it?" for any disease.
HARVEST_RULE = {
    "en": "Do not sell or eat fruit or tubers that show spots, rot or mould - remove them. "
          "Healthy-looking produce from sprayed plants may be harvested only after the waiting "
          "period on the product label (pre-harvest interval). Wash produce before eating.",
    "tl": "Wag ibenta o kainin yung bunga o patatas na may spots, bulok o amag - itapon na yun. "
          "Kung na-spray-an yung halaman, pwede lang i-harvest yung maayos na bunga after ng ilang "
          "araw na nakasulat sa label (pre-harvest interval). Hugasan muna bago kainin.",
}

SRC_UC = "UC IPM Pest Management Guidelines - University of California (ipm.ucanr.edu)"
SRC_CORNELL = "Vegetable MD Online - Cornell University (vegetablemdonline.ppath.cornell.edu)"
SRC_PSU = "Penn State Extension - vegetable and field crop disease fact sheets (extension.psu.edu)"
SRC_CIMMYT = "CIMMYT Maize Doctor / maize disease field guide (maizedoctor.org)"
SRC_CIP = "International Potato Center (CIP) - late blight management (cipotato.org)"
SRC_LOCAL = "Your Municipal Agriculture Office (MAO) / DA Regional Field Office / BPI"

_BLIGHT_FUNGICIDES = {
    "en": "Protectant fungicides with mancozeb or chlorothalonil; systemic options include "
          "azoxystrobin or difenoconazole. Alternate groups so the fungus does not become resistant.",
    "tl": "Fungicide na may mancozeb o chlorothalonil; may systemic options din tulad ng "
          "azoxystrobin o difenoconazole. I-rotate yung groups para hindi maging resistant yung "
          "fungus.",
}

_HEALTHY = {
    "cause": {"en": "No disease seen on this leaf.", "tl": "Walang nakitang sakit sa dahon na 'to."},
    "signs": {"en": "Even green colour, no spots, no curling.", "tl": "Pantay ang green, walang spots, hindi kulot."},
    "do_now": {"en": ["Keep checking a few plants every 2-3 days, especially after rain.",
                      "Photograph any leaf that changes and scan it again."],
               "tl": ["I-check ang ilang halaman every 2-3 days, lalo na after umulan.",
                      "Pag may dahon na nag-iba ang itsura, picture-an at i-scan ulit."]},
    "prevention": {"en": ["Water at the base, not over the leaves.", "Keep the field weeded and spaced for air flow.",
                          "Rotate crops of different families each season."],
                   "tl": ["Diligan sa may puno, wag sa dahon.", "Laging tanggalin ang damo at bigyan ng space ang bawat tanim para may hangin.",
                          "Mag-crop rotation every season - iba-ibang klase ng tanim."]},
    "chemical": None, "organic": None,
    "escalate": {"en": "Not needed.", "tl": "Hindi na kailangan."},
    "sources": [SRC_LOCAL],
}

TREATMENTS = {
    "Tomato___Bacterial_spot": {
        "cause": {"en": "Bacteria (Xanthomonas species), spread by rain splash, tools and infected seed.",
                  "tl": "Bacteria (Xanthomonas) - kumakalat through talsik ng ulan, mga gamit, at infected na "
                        "binhi."},
        "signs": {"en": "Small dark, water-soaked spots on leaves, often with a yellow edge; raised scabby spots on fruit.",
                  "tl": "Maliliit na dark spots sa dahon na parang basa, minsan may yellow na gilid; magaspang na "
                        "spots sa bunga."},
        "do_now": {"en": ["Remove badly spotted leaves and take them out of the field.",
                          "Do not work among plants while they are wet.",
                          "Clean tools and hands between rows."],
                   "tl": ["Tanggalin yung mga dahon na maraming spots at ilabas sa farm.",
                          "Wag galawin ang halaman habang basa pa.",
                          "Linisin ang tools at kamay bago lumipat sa ibang row."]},
        "prevention": {"en": ["Use certified clean seed or seedlings.", "Rotate away from tomato and pepper for 2-3 seasons.",
                              "Avoid overhead watering; use drip or water at the base."],
                       "tl": ["Gumamit ng certified na binhi o seedlings.", "Wag munang magtanim ng kamatis o sili sa same spot for 2-3 seasons.",
                              "Iwasan ang pagdidilig from above; drip o sa may puno na lang."]},
        "chemical": {"en": "Copper-based bactericides (e.g. copper hydroxide), often mixed with mancozeb, slow the spread. They protect new growth; they do not cure spotted leaves.",
                     "tl": "Copper-based na bactericide (e.g. copper hydroxide), madalas may halong mancozeb, para "
                           "bumagal ang pagkalat. Pang-protect ito sa bagong dahon - hindi nito napapagaling yung "
                           "may spots na."},
        "organic": {"en": "Copper products allowed in organic farming, plus strict sanitation.",
                    "tl": "Copper na approved for organic, plus laging linisin ang taniman."},
        "escalate": {"en": "If spots appear on most plants or on the fruit, bring a sample to the MAO.",
                     "tl": "Kung karamihan ng halaman o yung bunga ay may spots na, magdala ng sample sa MAO."},
        "sources": [SRC_UC, SRC_CORNELL, SRC_LOCAL],
    },
    "Tomato___Early_blight": {
        "cause": {"en": "Fungus (Alternaria species); survives on old plant debris and spreads in warm, humid weather.",
                  "tl": "Fungus (Alternaria) - nabubuhay sa mga lumang tira ng halaman at kumakalat pag mainit at "
                        "humid."},
        "signs": {"en": "Brown spots with rings like a target, starting on the lowest, oldest leaves, which then yellow.",
                  "tl": "Brown spots na may rings na parang target, nagsisimula sa pinakamababang dahon na "
                        "nagye-yellow."},
        "do_now": {"en": ["Strip the lower spotted leaves and remove them from the field.",
                          "Mulch the soil so rain does not splash spores up."],
                   "tl": ["Tanggalin yung mga dahon sa baba na may spots at ilabas sa farm.",
                          "Lagyan ng mulch ang lupa para hindi tumalsik ang spores."]},
        "prevention": {"en": ["Rotate away from tomato, potato and eggplant for 2-3 seasons.", "Plough under or remove crop debris after harvest.",
                              "Keep plants well fed - stressed plants get it worse."],
                       "tl": ["Wag munang magtanim ng kamatis, patatas o talong sa same spot for 2-3 seasons.", "Ibaon o tanggalin ang mga tira ng tanim after mag-harvest.",
                              "Lagyan ng tamang fertilizer - mas tinatamaan yung mahinang halaman."]},
        "chemical": _BLIGHT_FUNGICIDES,
        "organic": {"en": "Copper fungicides and Bacillus subtilis products give partial protection; sanitation matters most.",
                    "tl": "Copper at products na may Bacillus subtilis - may konting protection; pero "
                          "pinaka-importante pa rin ang kalinisan."},
        "escalate": {"en": "If it climbs past the middle of the plants within a week, ask the MAO.",
                     "tl": "Kung umakyat na lampas kalahati ng halaman within one week, magtanong sa MAO."},
        "sources": [SRC_UC, SRC_PSU, SRC_LOCAL],
    },
    "Tomato___Late_blight": {
        "cause": {"en": "Water mould (Phytophthora infestans). Spreads very fast in cool, wet weather and can kill a field in days.",
                  "tl": "Water mould (Phytophthora infestans). Sobrang bilis kumalat pag malamig at basa; kaya "
                        "nitong patayin yung buong taniman in a few days."},
        "signs": {"en": "Large, greasy grey-green to dark brown patches; white fuzz on the underside in humid mornings; firm brown rot on fruit.",
                  "tl": "Malalaking spots na parang basa, grayish-green hanggang dark; may puting parang bulak sa "
                        "ilalim ng dahon pag humid ang umaga; matigas na brown na bulok sa bunga."},
        "do_now": {"en": ["Act today: pull out and bag badly affected plants - do not compost them.",
                          "Protect the healthy plants around them with a fungicide right away.",
                          "Warn neighbouring farms; spores travel on the wind."],
                   "tl": ["Gawin agad today: bunutin at isako yung mga halamang malala na - wag i-compost.",
                          "I-spray agad ng fungicide yung healthy na halaman sa paligid para ma-protect.",
                          "Sabihan ang mga katabing farm - nadadala ng hangin yung spores."]},
        "prevention": {"en": ["Plant resistant varieties where available.", "Do not plant tomato next to potato.",
                              "Spray protectively before long cool, rainy spells (e.g. highland wet season)."],
                       "tl": ["Magtanim ng variety na resistant sa late blight kung meron.", "Wag pagtabihin ang kamatis at patatas.",
                              "Mag-spray as protection bago dumating ang mahabang malamig at maulang panahon (e.g. "
                              "rainy season sa bundok)."]},
        "chemical": {"en": "Protectants (mancozeb, chlorothalonil) before infection; for active outbreaks, products with cymoxanil, dimethomorph or metalaxyl-M combined with a protectant. Never rely on one product.",
                     "tl": "Protectant (mancozeb, chlorothalonil) bago pa tamaan; pag kumakalat na, cymoxanil, "
                           "dimethomorph o metalaxyl-M na may kasamang protectant. Wag umasa sa iisang product lang."},
        "organic": {"en": "Copper fungicides slow it but rarely stop an outbreak; removal of infected plants is essential.",
                    "tl": "Napapabagal ito ng copper pero bihirang napapatigil; kailangan talagang tanggalin yung "
                          "may sakit na halaman."},
        "escalate": {"en": "Report it to the MAO the same day - late blight affects the whole area.",
                     "tl": "I-report sa MAO the same day - apektado ang buong area pag may late blight."},
        "sources": [SRC_CIP, SRC_UC, SRC_CORNELL, SRC_LOCAL],
    },
    "Tomato___Leaf_Mold": {
        "cause": {"en": "Fungus (Passalora fulva), favoured by high humidity, especially in greenhouses and dense planting.",
                  "tl": "Fungus (Passalora fulva) - lumalala pag sobrang humid, lalo na sa greenhouse at siksik "
                        "na tanim."},
        "signs": {"en": "Pale yellow patches on the top of the leaf with olive-green to brown velvety mould underneath.",
                  "tl": "Pale yellow na spots sa ibabaw ng dahon, at parang velvet na amag na olive-green "
                        "hanggang brown sa ilalim."},
        "do_now": {"en": ["Open up the canopy: prune and remove the mouldy leaves.", "Increase ventilation; avoid wetting leaves."],
                   "tl": ["Bawasan ang siksik: putulin yung mga dahong may amag.", "Dagdagan ang airflow; wag basain ang dahon."]},
        "prevention": {"en": ["Wider spacing and staking.", "Water early in the day at the base.", "Resistant varieties where available."],
                       "tl": ["Lakihan ang space between plants at lagyan ng tukod (stake).", "Magdilig sa umaga at sa may puno.", "Gumamit ng resistant na variety kung meron."]},
        "chemical": {"en": "Chlorothalonil or mancozeb as protectants; some strobilurin products are labelled for it.",
                     "tl": "Chlorothalonil o mancozeb as protection; may ilang strobilurin din na naka-label para "
                           "dito."},
        "organic": {"en": "Ventilation and pruning do most of the work; copper gives some protection.",
                    "tl": "Malaking tulong ang good airflow at pag-prune ng sobrang dahon; may konting protection "
                          "din ang copper."},
        "escalate": {"en": "If it keeps spreading after improving airflow, ask the MAO.",
                     "tl": "Kung kumakalat pa rin kahit okay na ang airflow, magtanong sa MAO."},
        "sources": [SRC_UC, SRC_PSU, SRC_LOCAL],
    },
    "Tomato___Septoria_leaf_spot": {
        "cause": {"en": "Fungus (Septoria lycopersici), splashed up from soil and debris by rain.",
                  "tl": "Fungus (Septoria lycopersici) - tumatalsik galing sa lupa at tira ng halaman pag umuulan."},
        "signs": {"en": "Many small round spots with dark borders and pale grey centres, often with tiny black dots, on lower leaves first.",
                  "tl": "Maraming maliliit na bilog na spots - dark ang gilid, gray ang gitna, minsan may tiny "
                        "black dots - sa mga dahon sa baba muna."},
        "do_now": {"en": ["Remove spotted lower leaves.", "Mulch to stop soil splash.", "Avoid working among wet plants."],
                   "tl": ["Tanggalin yung mga dahon sa baba na may spots.", "Mag-mulch para hindi tumalsik ang lupa.", "Wag galawin ang halaman pag basa."]},
        "prevention": {"en": ["Rotate 2-3 seasons away from tomato.", "Clear and destroy crop debris.", "Control solanaceous weeds."],
                       "tl": ["Wag munang magkamatis sa same spot for 2-3 seasons.", "Linisin at itapon ang mga tira ng tanim.", "Tanggalin ang mga damong kamag-anak ng kamatis (same plant family)."]},
        "chemical": _BLIGHT_FUNGICIDES,
        "organic": {"en": "Copper fungicides plus sanitation.", "tl": "Copper na fungicide, plus kalinisan sa taniman."},
        "escalate": {"en": "If plants lose most lower leaves within two weeks, ask the MAO.",
                     "tl": "Kung nalagas na ang karamihan ng dahon sa baba within two weeks, magtanong sa MAO."},
        "sources": [SRC_CORNELL, SRC_PSU, SRC_LOCAL],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "cause": {"en": "Not a disease: tiny two-spotted spider mites (Tetranychus urticae), worst in hot, dry weather.",
                  "tl": "Hindi ito sakit: tiny pests na spider mites (Tetranychus urticae) - lumalala pag mainit "
                        "at tuyo."},
        "signs": {"en": "Fine pale speckling on leaves, bronzing, and fine webbing under the leaves.",
                  "tl": "Maliliit na white dots sa dahon, parang kinakalawang ang kulay, at may manipis na sapot "
                        "(webbing) sa ilalim."},
        "do_now": {"en": ["Check the leaf underside with a magnifier to confirm mites.", "Spray the undersides with a strong stream of water.",
                          "Remove heavily infested leaves."],
                   "tl": ["I-check ang ilalim ng dahon gamit ang magnifier para sure.", "I-spray ng malakas na tubig ang ilalim ng dahon.",
                          "Tanggalin yung mga dahon na puno na ng mites."]},
        "prevention": {"en": ["Avoid drought stress and dusty conditions.", "Avoid broad-spectrum insecticides - they kill the mites' natural enemies."],
                       "tl": ["Wag hayaang ma-dehydrate ang halaman; iwasan ang maalikabok na paligid.", "Iwasan ang insecticide na pumapatay ng lahat ng insects - namamatay din yung mga "
                                                                                                     "kumakain sa mites."]},
        "chemical": {"en": "Miticides (e.g. abamectin), rotated between different groups because mites become resistant quickly.",
                     "tl": "Miticide (e.g. abamectin) - i-rotate kasi mabilis maging resistant ang mites."},
        "organic": {"en": "Insecticidal soap or neem oil sprayed on leaf undersides, repeated per label.",
                    "tl": "Insecticidal soap o neem oil sa ilalim ng dahon, ulitin ayon sa label."},
        "escalate": {"en": "If webbing covers the tops of plants, ask the MAO.",
                     "tl": "Kung may sapot na hanggang tuktok ng halaman, magtanong sa MAO."},
        "sources": [SRC_UC, SRC_LOCAL],
    },
    "Tomato___Target_Spot": {
        "cause": {"en": "Fungus (Corynespora cassiicola), favoured by warm, humid weather and dense canopies.",
                  "tl": "Fungus (Corynespora cassiicola) - lumalala pag mainit, humid, at siksik ang tanim."},
        "signs": {"en": "Brown spots with light centres and faint rings; can merge and blight whole leaves.",
                  "tl": "Brown spots na light ang gitna at may malabong rings; nagdudugtong-dugtong hanggang "
                        "matuyo ang dahon."},
        "do_now": {"en": ["Remove affected leaves.", "Prune to open the canopy."],
                   "tl": ["Tanggalin yung mga apektadong dahon.", "Mag-prune ng ilang dahon para lumuwag at mahanginan ang tanim."]},
        "prevention": {"en": ["Rotate crops.", "Destroy debris after harvest.", "Avoid excess nitrogen that makes dense growth."],
                       "tl": ["Mag-crop rotation.", "Itapon ang mga tira after mag-harvest.", "Iwasan ang sobrang nitrogen - nagpapasiksik ito ng dahon."]},
        "chemical": {"en": "Chlorothalonil, mancozeb or azoxystrobin-type fungicides, alternated.",
                     "tl": "Chlorothalonil, mancozeb, o azoxystrobin - i-rotate."},
        "organic": {"en": "Sanitation and airflow; copper gives limited protection.",
                    "tl": "Kalinisan at good airflow; may konting protection ang copper."},
        "escalate": {"en": "If fruit starts to show sunken spots, ask the MAO.",
                     "tl": "Kung may lubog na spots na pati sa bunga, magtanong sa MAO."},
        "sources": [SRC_UC, SRC_LOCAL],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cause": {"en": "Virus (TYLCV) spread by whiteflies (Bemisia tabaci). There is no spray that cures it.",
                  "tl": "Virus (TYLCV) na dala ng whitefly (Bemisia tabaci). Walang spray na makakapagpagaling dito."},
        "signs": {"en": "Small leaves curling upward with yellow edges, stunted plants, few flowers and fruit.",
                  "tl": "Maliliit na dahon na kumukulot pataas na may yellow na gilid, halamang bansot (hindi "
                        "lumalaki), konti ang bulaklak at bunga."},
        "do_now": {"en": ["Pull out and bag infected plants early - they keep feeding the whiteflies the virus.",
                          "Hang yellow sticky traps to monitor and reduce whiteflies."],
                   "tl": ["Bunutin at isako agad yung may sakit na halaman - dito nakukuha ng whitefly ang virus.",
                          "Magsabit ng yellow sticky traps para ma-monitor at mabawasan ang whitefly."]},
        "prevention": {"en": ["Plant TYLCV-resistant varieties.", "Raise seedlings under insect netting.",
                              "Keep a break between tomato crops and remove weeds that host whiteflies."],
                       "tl": ["Magtanim ng variety na resistant sa TYLCV.", "Palakihin ang seedlings sa loob ng insect net.",
                              "Mag-break muna bago magtanim ulit ng kamatis, at tanggalin ang mga damong tinitirhan ng "
                              "whitefly."]},
        "chemical": {"en": "Insecticides only reduce the whitefly carrier, not the virus; rotate groups and follow the label.",
                     "tl": "Pambawas lang ng whitefly ang insecticide, hindi gamot sa virus; i-rotate ang groups at "
                           "sundin ang label."},
        "organic": {"en": "Netting, sticky traps, neem or soap sprays against whiteflies.",
                    "tl": "Net, sticky traps, at neem o soap spray laban sa whitefly."},
        "escalate": {"en": "Report spreading outbreaks to the MAO; whitefly control works best area-wide.",
                     "tl": "I-report sa MAO kung kumakalat; mas effective kung sabay-sabay ang buong area sa "
                           "pag-control ng whitefly."},
        "sources": [SRC_UC, SRC_CORNELL, SRC_LOCAL],
    },
    "Tomato___Tomato_mosaic_virus": {
        "cause": {"en": "Virus (ToMV) spread by touch - hands, tools, clothes - and by seed. Very stable; there is no cure.",
                  "tl": "Virus (ToMV) na kumakalat through hawak - kamay, tools, damit - at binhi. Matagal itong "
                        "nabubuhay; walang gamot dito."},
        "signs": {"en": "Light and dark green mottling, distorted or fern-like leaves, uneven ripening of fruit.",
                  "tl": "Halo-halong light at dark green na patches (mosaic), baluktot o makitid na dahon, hindi "
                        "pantay ang hinog ng bunga."},
        "do_now": {"en": ["Remove and bag infected plants.", "Wash hands with soap before touching other plants.",
                          "Disinfect pruning tools between plants."],
                   "tl": ["Tanggalin at isako yung may sakit na halaman.", "Maghugas ng kamay with soap bago humawak ng ibang halaman.",
                          "Linisin ang gunting/pruner bago lumipat sa ibang halaman."]},
        "prevention": {"en": ["Use certified seed and resistant varieties.", "No smoking or tobacco handling near plants.",
                              "Handle healthy plants before suspect ones."],
                       "tl": ["Gumamit ng certified na binhi at resistant na variety.", "Bawal mag-yosi o humawak ng tabako malapit sa tanim.",
                              "Hawakan muna yung healthy na halaman bago yung mukhang may sakit."]},
        "chemical": None,
        "organic": None,
        "escalate": {"en": "Ask the MAO to confirm - mosaic symptoms can also come from other viruses.",
                     "tl": "Ipa-check sa MAO - may ibang virus na kamukha nito."},
        "sources": [SRC_CORNELL, SRC_LOCAL],
    },
    "Tomato___healthy": _HEALTHY,
    "Pepper,_bell___Bacterial_spot": {
        "cause": {"en": "Bacteria (Xanthomonas species), carried on seed and spread by rain splash.",
                  "tl": "Bacteria (Xanthomonas) - dala ng binhi at kumakalat through talsik ng ulan."},
        "signs": {"en": "Small water-soaked spots that turn brown, yellowing and dropping leaves; raised spots on fruit.",
                  "tl": "Maliliit na spots na parang basa na nagiging brown, dahon na nagye-yellow at nalalagas; "
                        "nakaumbok na spots sa bunga."},
        "do_now": {"en": ["Remove spotted leaves and fallen leaves.", "Avoid handling wet plants."],
                   "tl": ["Tanggalin yung mga dahong may spots at yung mga nalagas na.", "Wag hawakan ang halaman pag basa."]},
        "prevention": {"en": ["Certified seed and resistant varieties.", "Rotate away from pepper and tomato for 2-3 seasons.",
                              "Drip or base watering."],
                       "tl": ["Certified na binhi at resistant na variety.", "Wag munang magsili o magkamatis sa same spot for 2-3 seasons.",
                              "Magdilig sa may puno."]},
        "chemical": {"en": "Copper bactericides, often with mancozeb, to protect new growth.",
                     "tl": "Copper na bactericide, madalas may mancozeb, para ma-protect ang bagong dahon."},
        "organic": {"en": "Organic-approved copper plus sanitation.", "tl": "Copper na approved for organic, plus kalinisan."},
        "escalate": {"en": "If leaf drop is heavy, bring a sample to the MAO.",
                     "tl": "Kung marami nang nalalagas na dahon, magdala ng sample sa MAO."},
        "sources": [SRC_UC, SRC_CORNELL, SRC_LOCAL],
    },
    "Pepper,_bell___healthy": _HEALTHY,
    "Potato___Early_blight": {
        "cause": {"en": "Fungus (Alternaria solani), worst on older or stressed plants in warm, humid weather.",
                  "tl": "Fungus (Alternaria solani) - lumalala sa matanda o mahinang halaman pag mainit at humid."},
        "signs": {"en": "Dark brown spots with target-like rings on older leaves, often bordered by yellow.",
                  "tl": "Dark brown spots na parang target sa mga lumang dahon, madalas may yellow na gilid."},
        "do_now": {"en": ["Remove the worst leaves.", "Keep plants watered and fertilised to reduce stress."],
                   "tl": ["Tanggalin yung mga pinakamalalang dahon.", "Diligan at lagyan ng tamang fertilizer para hindi manghina."]},
        "prevention": {"en": ["Rotate away from potato, tomato and eggplant.", "Remove volunteer potatoes and debris."],
                       "tl": ["Wag magpatatas, magkamatis o magtalong ulit sa same spot.", "Tanggalin ang mga kusang tumubong patatas at mga tira ng tanim."]},
        "chemical": _BLIGHT_FUNGICIDES,
        "organic": {"en": "Copper and Bacillus-based products; good nutrition and rotation.",
                    "tl": "Copper at products na may Bacillus; tamang fertilizer at crop rotation."},
        "escalate": {"en": "If spots move onto the upper leaves before tubers bulk, ask the MAO.",
                     "tl": "Kung umaakyat na sa mga dahon sa taas bago pa lumaki ang patatas, magtanong sa MAO."},
        "sources": [SRC_UC, SRC_PSU, SRC_LOCAL],
    },
    "Potato___Late_blight": {
        "cause": {"en": "Water mould (Phytophthora infestans) - the disease behind the Irish famine. Spreads in days in cool, wet weather; tubers can rot in storage.",
                  "tl": "Water mould (Phytophthora infestans). Kumakalat in a few days pag malamig at basa; "
                        "nabubulok din ang patatas kahit naka-storage na."},
        "signs": {"en": "Dark, water-soaked patches, white mould on the underside in humid weather, blackened stems; reddish-brown rot inside tubers.",
                  "tl": "Dark spots na parang basa, puting amag sa ilalim pag humid, nangingitim na tangkay; "
                        "reddish-brown na bulok sa loob ng patatas."},
        "do_now": {"en": ["Act today: remove and bag infected plants.", "Spray the healthy part of the field to protect it.",
                          "Hill up soil over tubers so spores do not wash down to them."],
                   "tl": ["Gawin agad today: tanggalin at isako yung may sakit na halaman.", "I-spray yung healthy na part ng taniman as protection.",
                          "Tabunan ng lupa ang puno (hilling) para hindi madala ng tubig ang spores papunta sa "
                          "patatas."]},
        "prevention": {"en": ["Plant certified, disease-free seed tubers and resistant varieties.", "Destroy cull piles and volunteers.",
                              "Spray protectively before cool, rainy spells."],
                       "tl": ["Gumamit ng certified na seed potatoes at resistant na variety.", "Itapon ang tambak ng reject na patatas at bunutin ang mga kusang tumubo.",
                              "Mag-spray as protection bago dumating ang malamig at maulang panahon."]},
        "chemical": {"en": "Protectants (mancozeb, chlorothalonil) before infection; cymoxanil, dimethomorph or metalaxyl-M mixtures once it is present. Alternate groups.",
                     "tl": "Protectant (mancozeb, chlorothalonil) bago pa tamaan; pag meron na, mix na may "
                           "cymoxanil, dimethomorph o metalaxyl-M. I-rotate ang groups."},
        "organic": {"en": "Copper slows it; removal and resistant varieties are essential.",
                    "tl": "Napapabagal ng copper; kailangan pa rin tanggalin ang may sakit at gumamit ng resistant "
                          "na variety."},
        "escalate": {"en": "Report to the MAO the same day.", "tl": "I-report sa MAO the same day."},
        "sources": [SRC_CIP, SRC_UC, SRC_LOCAL],
    },
    "Potato___healthy": _HEALTHY,
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "cause": {"en": "Fungus (Cercospora zeae-maydis), surviving on corn residue; favoured by long humid, warm periods.",
                  "tl": "Fungus (Cercospora zeae-maydis) - nabubuhay sa tira ng mais; lumalala pag matagal na "
                        "mainit at humid."},
        "signs": {"en": "Long, narrow, rectangular grey to tan spots running between the leaf veins.",
                  "tl": "Mahahaba at makikitid na spots (parang rectangle), gray hanggang brown, sa pagitan ng "
                        "mga ugat ng dahon."},
        "do_now": {"en": ["Check how high on the plant it has reached; the leaves above the ear matter most for yield."],
                   "tl": ["I-check kung gaano kataas na ang sakit; pinaka-importante sa harvest yung mga dahon sa "
                          "taas ng bunga."]},
        "prevention": {"en": ["Plant tolerant hybrids.", "Rotate away from corn for a season.", "Bury or remove residue."],
                       "tl": ["Magtanim ng hybrid na resistant dito.", "Wag magmais ulit sa same spot sa susunod na season.", "Ibaon o tanggalin ang mga tira."]},
        "chemical": {"en": "Strobilurin and/or triazole fungicides around tasselling, if the disease is climbing toward the ear leaf.",
                     "tl": "Strobilurin at/o triazole na fungicide around flowering (paglabas ng tassel), kung "
                           "umaakyat na papunta sa dahon ng bunga."},
        "organic": {"en": "Rotation, residue management and tolerant hybrids.", "tl": "Crop rotation, paglilinis ng tira, at resistant na hybrid."},
        "escalate": {"en": "Ask the MAO whether a spray is worth it at your growth stage.",
                     "tl": "Itanong sa MAO kung worth it pa mag-spray sa edad ng mais n'yo."},
        "sources": [SRC_CIMMYT, SRC_PSU, SRC_LOCAL],
    },
    "Corn___Common_rust": {
        "cause": {"en": "Fungus (Puccinia sorghi), blown in by wind; favoured by cool nights and dew.",
                  "tl": "Fungus (Puccinia sorghi) - dala ng hangin; lumalala pag malamig ang gabi at may hamog."},
        "signs": {"en": "Small, raised, cinnamon-brown powdery pustules on both sides of the leaf.",
                  "tl": "Maliliit na nakaumbok na butlig, parang pulbos at kulay-kalawang, sa harap at likod ng "
                        "dahon."},
        "do_now": {"en": ["Usually no action on field corn if it appears late; watch the leaves above the ear."],
                   "tl": ["Usually wala nang kailangang gawin kung late na lumabas; bantayan lang yung mga dahon sa "
                          "taas ng bunga."]},
        "prevention": {"en": ["Plant resistant hybrids.", "Avoid very late planting when rust pressure is high."],
                       "tl": ["Magtanim ng resistant na hybrid.", "Wag masyadong i-delay ang pagtanim pag maraming rust sa area."]},
        "chemical": {"en": "Triazole or strobilurin fungicides, mainly for sweet corn or susceptible hybrids with early, heavy infection.",
                     "tl": "Triazole o strobilurin, lalo na sa sweet corn o mahinang hybrid na maaga at malala ang tama."},
        "organic": {"en": "Resistant hybrids are the main tool.", "tl": "Resistant na hybrid ang pinaka-effective na paraan."},
        "escalate": {"en": "If pustules cover most of the upper leaves before tasselling, ask the MAO.",
                     "tl": "Kung puno na ng butlig ang mga dahon sa taas bago pa mag-flowering, magtanong sa MAO."},
        "sources": [SRC_CIMMYT, SRC_PSU, SRC_LOCAL],
    },
    "Corn___Northern_Leaf_Blight": {
        "cause": {"en": "Fungus (Exserohilum turcicum), surviving on residue; favoured by moderate temperatures and long dew periods.",
                  "tl": "Fungus (Exserohilum turcicum) - nabubuhay sa tira; lumalala pag hindi masyadong mainit "
                        "at matagal ang hamog."},
        "signs": {"en": "Long, cigar-shaped grey-green to tan lesions, several centimetres long.",
                  "tl": "Mahahabang spots na hugis-tabako (cigar-shaped), grayish-green hanggang brown, ilang "
                        "centimeters ang haba."},
        "do_now": {"en": ["Check whether lesions are on the ear leaf or above before deciding on a spray."],
                   "tl": ["I-check kung nasa dahon ng bunga o mas mataas na yung spots bago mag-decide mag-spray."]},
        "prevention": {"en": ["Resistant hybrids.", "Rotation and residue management."],
                       "tl": ["Resistant na hybrid.", "Crop rotation at paglilinis ng tira."]},
        "chemical": {"en": "Strobilurin and/or triazole fungicides around tasselling when lesions reach the ear leaf on susceptible hybrids.",
                     "tl": "Strobilurin at/o triazole around flowering, pag umabot na sa dahon ng bunga sa mahinang "
                           "hybrid."},
        "organic": {"en": "Rotation, residue management and resistant hybrids.", "tl": "Crop rotation, paglilinis ng tira, at resistant na hybrid."},
        "escalate": {"en": "Ask the MAO whether a spray pays at your stage.", "tl": "Itanong sa MAO kung worth it ang spray sa edad ng mais n'yo."},
        "sources": [SRC_CIMMYT, SRC_PSU, SRC_LOCAL],
    },
    "Corn___healthy": _HEALTHY,
}
