"""
Treatment Guidance Generation
Uses LLM or template-based system for treatment recommendations
"""
import os

# Template-based guidance (fallback when LLM is unavailable)
TREATMENT_TEMPLATES = {
    'Tomato Late Blight': {
        'en': '''**Treatment for Late Blight (Tomato)**

1. **Remove Infected Leaves**: Cut and destroy all affected leaves immediately. Do not compost them.

2. **Apply Fungicide**: Use copper-based fungicide (e.g., Kocide, Cupravit) or mancozeb. Spray every 7 days.
   - Dosage: Follow label instructions (typically 2-3g per liter of water)
   - Apply early morning or late afternoon

3. **Improve Air Circulation**: Space plants wider apart, prune lower leaves

4. **Reduce Humidity**: Avoid overhead watering. Water at base of plant only.

5. **Organic Alternative**: Baking soda spray (1 tablespoon per gallon water + few drops dish soap)

**Prevention**: Plant resistant varieties, rotate crops yearly, avoid wetting leaves when watering.

⚠️ **Important**: Do not harvest within 7 days of fungicide application. Wash thoroughly before eating.''',

        'tl': '''**Paggamot para sa Late Blight (Kamatis)**

1. **Tanggalin ang May Sakit na Dahon**: Putulin at itapon kaagad ang lahat ng apektadong dahon. Huwag i-compost.

2. **Magspray ng Fungicide**: Gumamit ng copper-based fungicide (hal. Kocide, Cupravit) o mancozeb. Mag-spray tuwing 7 araw.
   - Dosis: Sundin ang nakasulat sa label (karaniwang 2-3g bawat litro ng tubig)
   - Mag-spray ng maaga sa umaga o hapon

3. **Pagandahin ang Daloy ng Hangin**: Palambutin ang pagkakatayo ng halaman, putol ang ibabang dahon

4. **Bawasan ang Halumigmig**: Iwasan ang pagdidilig sa ibabaw. Sa ugat lang ng halaman magdilig.

5. **Organic na Paraan**: Baking soda spray (1 kutsara bawat galon ng tubig + ilang patak ng sabon)

**Pag-iwas**: Magtanim ng resistant varieties, mag-rotate ng pananim bawat taon, iwasan ang pagbasa ng dahon.

⚠️ **Mahalaga**: Huwag mag-ani hanggang 7 araw pagkatapos ng fungicide. Hugasan ng mabuti bago kainin.'''
    },

    'Tomato Early Blight': {
        'en': '''**Treatment for Early Blight (Tomato)**

1. **Remove Lower Leaves**: Remove affected leaves and bottom 6 inches of plant

2. **Apply Fungicide**: Chlorothalonil or copper fungicide every 7-10 days
   - Start treatment at first sign of symptoms

3. **Mulch**: Apply organic mulch to prevent soil splash onto leaves

4. **Water Management**: Water at soil level, avoid wetting foliage

5. **Organic Option**: Neem oil spray weekly

**Prevention**: Use disease-free transplants, stake plants for air flow, rotate with non-solanaceous crops.''',

        'tl': '''**Paggamot para sa Early Blight (Kamatis)**

1. **Tanggalin ang Ibabang Dahon**: Alisin ang may sakit na dahon at 6 pulgada sa ibaba ng halaman

2. **Magspray ng Fungicide**: Chlorothalonil o copper fungicide tuwing 7-10 araw
   - Simulan ang treatment sa unang senyales

3. **Mulch**: Lagyan ng organic mulch para hindi tumapon ang lupa sa dahon

4. **Pag-aalaga sa Tubig**: Sa lupa lang magdilig, iwasan ang dahon

5. **Organic**: Neem oil spray bawat linggo

**Pag-iwas**: Gumamit ng walang sakit na punla, ipatayo ang halaman, mag-rotate ng pananim.'''
    },

    'Pepper Bacterial Spot': {
        'en': '''**Treatment for Bacterial Spot (Pepper)**

1. **Remove Infected Parts**: Prune and destroy all spotted leaves

2. **Copper Spray**: Fixed copper bactericide every 5-7 days
   - Essential during wet weather

3. **Reduce Leaf Wetness**: Drip irrigation only, no overhead watering

4. **Sanitize Tools**: Disinfect pruning tools with bleach solution after each cut

5. **Spacing**: Ensure good air circulation between plants

**Prevention**: Use certified disease-free seeds, avoid working with wet plants, crop rotation.

⚠️ Bacterial diseases are hard to cure. Prevention is critical.''',

        'tl': '''**Paggamot para sa Bacterial Spot (Sili)**

1. **Tanggalin ang May Sakit**: Putulin at itapon ang lahat ng may batik na dahon

2. **Copper Spray**: Fixed copper bactericide tuwing 5-7 araw
   - Kailangan lalo na pag umuulan

3. **Bawasan ang Pagkabasa ng Dahon**: Drip irrigation lang, huwag sa ibabaw magdilig

4. **Linisin ang Gamit**: Hugasan ng bleach solution ang pangputol pagkatapos gamitin

5. **Pagitan**: Siguraduhing may sapat na hangin sa pagitan ng halaman

**Pag-iwas**: Gumamit ng certified seeds, iwasan ang paghawak ng basang halaman, mag-rotate ng tanim.

⚠️ Mahirap gamutin ang bacterial disease. Mahalaga ang pag-iwas.'''
    },

    'Rice Blast': {
        'en': '''**Treatment for Rice Blast**

1. **Apply Fungicide**: Tricyclazole or azoxystrobin at early disease stage
   - Follow recommended dosage per hectare

2. **Improve Drainage**: Ensure proper water management, avoid continuous flooding

3. **Balanced Fertilization**: Reduce excess nitrogen, add potassium and silica

4. **Remove Infected Plants**: In severe cases, remove heavily infected hills

5. **Spacing**: Maintain recommended plant spacing for air flow

**Prevention**: Use resistant varieties (ask PhilRice for recommendations), clean field after harvest.

Contact your local agricultural technician for specific fungicide recommendations.''',

        'tl': '''**Paggamot para sa Rice Blast (Palay)**

1. **Magspray ng Fungicide**: Tricyclazole o azoxystrobin sa maagang yugto
   - Sundin ang tamang dosis bawat ektarya

2. **Pagandahin ang Drainage**: Siguraduhing maayos ang tubig, iwasan ang sobrang baha

3. **Balanseng Pataba**: Bawasan ang sobrang nitrogen, dagdagan ng potassium at silica

4. **Tanggalin ang May Sakit**: Kung malala, alisin ang lubhang may sakit na punla

5. **Pagitan**: Panatilihin ang tamang agwat ng pananim para sa hangin

**Pag-iwas**: Gumamit ng resistant varieties (magtanong sa PhilRice), linisin ang bukid pagkatapos ng ani.

Makipag-ugnayan sa inyong agricultural technician para sa tiyak na rekomendasyon.'''
    }
}

def get_treatment_guidance(disease_name, crop_type, language='tl'):
    """
    Get treatment guidance for detected disease

    Args:
        disease_name: Name of detected disease
        crop_type: Type of crop (tomato, pepper, rice, etc.)
        language: 'en' or 'tl'

    Returns:
        dict with 'en' and 'tl' keys containing treatment text
    """

    # Check if we have a template for this disease
    if disease_name in TREATMENT_TEMPLATES:
        return TREATMENT_TEMPLATES[disease_name]

    # Generic fallback for diseases without specific templates
    generic = {
        'en': f'''**Treatment for {disease_name}**

1. **Isolate**: Remove and destroy infected plant parts immediately
2. **Fungicide/Pesticide**: Consult your local agricultural office for specific treatment
3. **Sanitation**: Clean tools and equipment after handling infected plants
4. **Monitor**: Check plants daily for spread
5. **Prevention**: Improve air circulation, reduce humidity, practice crop rotation

⚠️ For specific treatment recommendations, please consult with your local agricultural technician or extension office.''',

        'tl': f'''**Paggamot para sa {disease_name}**

1. **Ihiwalay**: Tanggalin at sirain kaagad ang may sakit na bahagi ng halaman
2. **Fungicide/Pesticide**: Kumonsulta sa inyong lokal na tanggapan ng agrikultura para sa tiyak na gamot
3. **Kalinisan**: Linisin ang mga gamit pagkatapos hawakan ang may sakit na halaman
4. **Bantayan**: Suriin araw-araw kung kumakalat
5. **Pag-iwas**: Pagandahin ang daloy ng hangin, bawasan ang halumigmig, mag-rotate ng pananim

⚠️ Para sa tiyak na rekomendasyon, mangyaring kumonsulta sa inyong lokal na agricultural technician o extension office.'''
    }

    return generic

def generate_llm_guidance(disease_name, crop_type, language='tl'):
    """
    Generate treatment guidance using LLM (future enhancement)

    This would call Ollama or OpenAI API to generate context-aware,
    localized treatment recommendations grounded in Philippine
    agricultural guidelines.

    For now, falls back to templates.
    """
    # TODO: Implement LLM integration
    # Could use Ollama (local, like carscoot_pos assistant)
    # or OpenAI API with prompt engineering

    return get_treatment_guidance(disease_name, crop_type, language)
