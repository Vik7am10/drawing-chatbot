#!/usr/bin/env python3
"""
Simple FastAPI backend using the working ArchRAGSystem + OpenAI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import openai
import os
import json
import re

# Import the working RAG system (bypasses dependency issues)
try:
    from intelligent_arch_system import IntelligentArchSystem
    HAS_INTELLIGENT = True
    print("✅ Using IntelligentArchSystem")
except ImportError:
    HAS_INTELLIGENT = False
    try:
        from arch_rag_system import ArchRAGSystem
        HAS_BASIC_RAG = True
        print("✅ Using basic ArchRAGSystem")
    except ImportError:
        HAS_BASIC_RAG = False
        print("❌ No RAG system available")

# Load OpenAI API key
try:
    with open('api_key', 'r') as f:
        openai.api_key = f.read().strip()
    print(f"✅ OpenAI API key loaded")
except FileNotFoundError:
    openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI(title="Fresco Architectural RAG API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    context_summary: str
    source_sheets: List[str]
    measurements: Optional[List[str]] = None
    parsed_query: Optional[Dict[str, Any]] = None
    image_data: Optional[str] = None
    image_filename: Optional[str] = None

# Initialize RAG system
rag_system = None
try:
    if HAS_INTELLIGENT:
        rag_system = IntelligentArchSystem()
        print("✅ IntelligentArchSystem initialized")
    elif HAS_BASIC_RAG:
        rag_system = ArchRAGSystem()
        print("✅ ArchRAGSystem initialized")
except Exception as e:
    print(f"❌ RAG system initialization failed: {e}")

def detect_document_query(query: str) -> Dict[str, any]:
    """Detect if query is asking about specific document(s) and analyze query type"""
    import re
    
    detected_sheets = []
    query_upper = query.upper()
    query_lower = query.lower()
    
    # First, look for full sheet names with '_-_' pattern (highest priority)
    full_sheet_pattern = r'[A-Z]\d*\.?\d*(?:\.\d+)?_-_[A-Z0-9_\-\s]+_\d+'
    full_matches = re.findall(full_sheet_pattern, query_upper)
    if full_matches:
        detected_sheets.extend(full_matches)
    else:
        # Look for sheet patterns like A8.4, C400, M-2.1, etc.
        sheet_patterns = [
            r'[A-Z]\d+\.?\d*\.?\d*',  # A8.4, C400, G1.1.1, G1.2
            r'[A-Z]-\d+\.?\d*',       # M-2.1, P-4.1
            r'FP-\d+\.?\d*',          # FP-4.0
            r'S\d+\.?\d*'             # S7.11
        ]
        
        # Find all sheet references in the query
        for pattern in sheet_patterns:
            matches = re.findall(pattern, query_upper)
            # Filter out false positives by checking context
            for match in matches:
                match_pos = query_upper.find(match)
                preceding_text = query_upper[:match_pos].split()
                following_pos = match_pos + len(match)
                following_text = query_upper[following_pos:following_pos+10].strip()
                
                # Include if it's in a reasonable context
                if (not preceding_text or  # At beginning
                    preceding_text[-1] in ['SHEET', 'DOCUMENT', 'IN', 'ON', 'FROM', 'OF'] or  # After sheet reference words
                    following_text.startswith(('_-_', '.', '?', ' ')) or  # Before sheet indicators
                    any(word in query_lower for word in ['analyze', 'what', 'how', 'show', 'tell', 'compare'])):  # Command context
                    detected_sheets.append(match)
    
    # Remove duplicates and limit to unique sheets
    unique_sheets = list(set(detected_sheets))
    
    # Determine query type - if we found cross-document keywords, it's cross-document even if sheets aren't detected yet
    cross_doc_keywords = [word for word in ['compare', 'difference', 'versus', 'vs', 'between', 'relationship', 'across', 'both'] if word in query_lower]
    is_cross_document = len(unique_sheets) > 1 or (len(cross_doc_keywords) > 0 and len(unique_sheets) >= 0)
    
    return {
        'sheets': unique_sheets,
        'count': len(unique_sheets),
        'is_cross_document': is_cross_document,
        'is_single_document': len(unique_sheets) == 1 and not is_cross_document,
        'cross_document_keywords': cross_doc_keywords
    }

def expand_sheet_names(short_names: List[str]) -> List[str]:
    """Expand short sheet names like 'A8.4' to full names like 'A8.4_-_INTERIOR_ELEVATIONS...'"""
    if not rag_system:
        return short_names
    
    try:
        # Get all available sheets from the database
        if HAS_INTELLIGENT:
            inventory = rag_system.rag_system.get_drawing_inventory()
        else:
            inventory = rag_system.get_drawing_inventory()
        
        all_sheets = list(inventory.get('sheets_detail', {}).keys())
        expanded_names = []
        
        for short_name in short_names:
            # Find matching full sheet names
            matches = []
            short_upper = short_name.upper()
            
            for full_sheet in all_sheets:
                if full_sheet.upper().startswith(short_upper + '_'):
                    matches.append(full_sheet)
            
            if matches:
                # If multiple matches, prefer the one without additional suffixes (like .1)
                if len(matches) > 1:
                    # Sort by length to prefer shorter (base) versions
                    matches.sort(key=len)
                expanded_names.extend(matches[:2])  # Max 2 to avoid too many matches
            else:
                # If no expansion found, keep the original
                expanded_names.append(short_name)
        
        return expanded_names
        
    except Exception as e:
        print(f"Error expanding sheet names: {e}")
        return short_names

def get_all_document_elements(sheet_id: str) -> Dict[str, Any]:
    """Get ALL elements from a specific document/sheet"""
    if not rag_system:
        return {"error": "RAG system not available"}
    
    try:
        if HAS_INTELLIGENT:
            # Use intelligent system to get comprehensive document analysis
            inventory = rag_system.rag_system.get_drawing_inventory()
            
            # Direct ChromaDB search - much simpler and more accurate!
            text_collection = rag_system.rag_system.text_collection
            image_collection = rag_system.rag_system.image_collection
            
            # Get ALL elements and filter by sheet_id in metadata
            text_results = text_collection.get(include=['metadatas', 'documents'])
            image_results = image_collection.get(include=['metadatas', 'documents'])
            
            # Filter for exact sheet_id match
            sheet_id_upper = sheet_id.upper()
            filtered_text = {"documents": [], "metadatas": []}
            filtered_images = {"documents": [], "metadatas": []}
            matching_sheet_id = None
            
            # Filter text results for exact match
            for doc, meta in zip(text_results['documents'], text_results['metadatas']):
                if sheet_id_upper in meta.get('sheet_id', '').upper():
                    filtered_text['documents'].append(doc)
                    filtered_text['metadatas'].append(meta)
                    if not matching_sheet_id:
                        matching_sheet_id = meta.get('sheet_id')
            
            # Filter image results for exact match
            for doc, meta in zip(image_results['documents'], image_results['metadatas']):
                if sheet_id_upper in meta.get('sheet_id', '').upper():
                    filtered_images['documents'].append(doc)
                    filtered_images['metadatas'].append(meta)
                    if not matching_sheet_id:
                        matching_sheet_id = meta.get('sheet_id')
            
            if not matching_sheet_id:
                return {"error": f"Sheet {sheet_id} not found in database"}
            
            text_results = filtered_text
            image_results = filtered_images
            
            return {
                "sheet_id": matching_sheet_id,
                "text_elements": text_results,
                "image_elements": image_results,
                "total_elements": len(text_results['documents']) + len(image_results['documents'])
            }
        else:
            return {"error": "Document-specific search requires intelligent system"}
            
    except Exception as e:
        return {"error": f"Failed to get document elements: {e}"}

def search_dimension_texts(dimension_elements: List[Tuple], query: str) -> List[Dict[str, Any]]:
    """Search dimension_line text elements semantically for relevant dimensions"""
    if not dimension_elements:
        return []
    
    try:
        # Simple relevance scoring based on query keywords
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        relevant_dimensions = []
        
        for doc, meta in dimension_elements:
            doc_lower = doc.lower()
            
            # Calculate relevance score
            doc_words = set(doc_lower.split())
            common_words = query_words.intersection(doc_words)
            relevance_score = len(common_words) / max(len(query_words), 1)
            
            # Include dimensions with any relevance or containing numbers
            if relevance_score > 0 or re.search(r'\d+', doc):
                relevant_dimensions.append({
                    'text': doc,
                    'metadata': meta,
                    'relevance_score': relevance_score,
                    'sheet_id': meta.get('sheet_id', ''),
                    'bbox': meta.get('bbox', None),
                    'element_type': meta.get('element_type', 'dimension_line')
                })
        
        # Sort by relevance score and return top matches
        relevant_dimensions.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_dimensions[:10]  # Limit to top 10
        
    except Exception as e:
        print(f"Error in dimension text search: {e}")
        return []

def interpret_dimensions_with_llm(dimension_texts: List[Dict], query: str, sheet_id: str) -> Dict[str, Any]:
    """Use LLM to interpret dimension texts and extract meaningful measurements"""
    if not dimension_texts:
        return {"measurements": [], "interpretation": "No dimension texts found"}
    
    try:
        # Prepare dimension texts for LLM
        dimension_context = []
        for i, dim in enumerate(dimension_texts):
            text = dim['text']
            bbox = dim['bbox']
            location_info = f" (at {bbox})" if bbox else ""
            dimension_context.append(f"Dimension {i+1}{location_info}: {text}")
        
        dimension_text_block = "\n".join(dimension_context)
        
        # Simple prompt for dimension interpretation
        prompt = f"""Answer this query using the dimension information from drawing {sheet_id}:

QUERY: {query}

DIMENSION INFORMATION:
{dimension_text_block}

Answer the query based on these dimensions."""

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an architectural consultant interpreting dimension texts from construction drawings."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        interpretation = response.choices[0].message.content.strip()
        
        # Return raw dimension texts for API response - let LLM handle interpretation
        measurement_values = []
        for dim in dimension_texts:
            # Return the full raw text from dimension_line, not extracted measurements
            measurement_values.append(dim['text'])
        
        return {
            "measurements": measurement_values,
            "interpretation": interpretation,
            "dimension_count": len(dimension_texts),
        }
        
    except Exception as e:
        return {"measurements": [], "interpretation": f"Error interpreting dimensions: {e}"}

def classify_query_intent(query: str) -> Dict[str, Any]:
    """Enhanced query classification for architectural workflows"""
    query_lower = query.lower()
    
    intent_patterns = {
        'measurement': {
            'keywords': ['height', 'width', 'dimension', 'size', 'feet', 'inches', 'mm', 'cm', 'meters', 'length', 'depth'],
            'requires_vision': True,  # Often need to see actual drawings for measurements
            'priority': 'high'
        },
        'count': {
            'keywords': ['how many', 'count', 'number of', 'total', 'quantity'],
            'requires_vision': True,  # Need to visually count objects
            'priority': 'high'
        },
        'description': {
            'keywords': ['describe', 'what is', 'tell me about', 'explain', 'appearance', 'look like'],
            'requires_vision': False,  # Can often get from text embeddings
            'priority': 'medium'
        },
        'location': {
            'keywords': ['where', 'located', 'position', 'find', 'placement'],
            'requires_vision': False,  # Text often has location info
            'priority': 'medium'
        },
        'material': {
            'keywords': ['material', 'finish', 'construction', 'made of', 'type of', 'specification'],
            'requires_vision': False,  # Usually in text specs
            'priority': 'medium'
        },
        'annotation': {
            'keywords': ['note', 'annotation', 'label', 'text', 'written', 'says', 'marked'],
            'requires_vision': False,  # Text embeddings capture annotations
            'priority': 'low'
        },
        'visual_identification': {
            'keywords': ['show', 'drawing', 'elevation', 'section', 'plan', 'view', 'layout', 'analyze', 'analysis'],
            'requires_vision': True,  # Need to see the actual drawings
            'priority': 'high'
        },
        'spatial': {
            'keywords': ['above', 'below', 'adjacent', 'next to', 'near', 'relationship', 'connection'],
            'requires_vision': True,  # Need spatial understanding
            'priority': 'high'
        }
    }
    
    detected_intents = []
    vision_required = False
    max_priority = 'low'
    
    for intent, config in intent_patterns.items():
        matches = [kw for kw in config['keywords'] if kw in query_lower]
        if matches:
            detected_intents.append({
                'intent': intent,
                'confidence': len(matches) / len(config['keywords']),
                'requires_vision': config['requires_vision'],
                'priority': config['priority'],
                'matched_keywords': matches
            })
            
            if config['requires_vision']:
                vision_required = True
            
            if config['priority'] == 'high':
                max_priority = 'high'
            elif config['priority'] == 'medium' and max_priority != 'high':
                max_priority = 'medium'
    
    # Special overrides for complex cases
    if any(word in query_lower for word in ['ceiling height', 'floor height', 'room height']):
        vision_required = True  # Heights often need visual confirmation
    
    if any(word in query_lower for word in ['all notes', 'annotations', 'text on']):
        vision_required = False  # Pure text extraction
    
    # Force vision for "show" commands with documents
    if query_lower.startswith('show ') or ' show ' in query_lower:
        vision_required = True  # Show commands need visual display
    
    return {
        'detected_intents': detected_intents,
        'requires_vision': vision_required,
        'priority': max_priority,
        'primary_intent': detected_intents[0]['intent'] if detected_intents else 'general',
        'confidence': max([intent['confidence'] for intent in detected_intents]) if detected_intents else 0.0
    }

def parse_query_with_openai(query: str) -> Dict[str, Any]:
    """Parse query using OpenAI"""
    prompt = f"""You are an expert architectural drawing analyst. Parse this query:

Query: "{query}"

Respond with ONLY valid JSON:
{{
  "intent_type": "dimensions|materials|specifications|structural|electrical|mechanical|general|inventory",
  "confidence": "high|medium|low", 
  "key_elements": ["list", "of", "terms"],
  "search_focus": "text_heavy|image_heavy|balanced",
  "expected_answer_type": "measurements|specifications|locations|descriptions|lists|count",
  "architectural_domain": "structural|architectural|mechanical|electrical|civil"
}}"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        return json.loads(response.choices[0].message.content.strip())
    except:
        return {
            "intent_type": "general",
            "confidence": "medium",
            "key_elements": query.split()[:5],
            "search_focus": "balanced",
            "expected_answer_type": "descriptions",
            "architectural_domain": "architectural"
        }

def get_relevant_bounding_boxes(document_data: Dict, query: str, intent_info: Dict) -> List[Dict]:
    """Extract bounding boxes of relevant regions for drawing highlighting"""
    relevant_boxes = []
    primary_intent = intent_info['primary_intent']
    
    # Check both text and image elements for spatial data
    all_elements = []
    
    # Add text elements
    for doc, meta in zip(document_data['text_elements']['documents'], 
                        document_data['text_elements']['metadatas']):
        all_elements.append({'content': doc, 'metadata': meta, 'source': 'text'})
    
    # Add image elements  
    for doc, meta in zip(document_data['image_elements']['documents'],
                        document_data['image_elements']['metadatas']):
        all_elements.append({'content': doc, 'metadata': meta, 'source': 'image'})
    
    query_lower = query.lower()
    
    for element in all_elements:
        meta = element['metadata']
        content = element['content'].lower()
        
        # Check if element is relevant to query
        is_relevant = False
        relevance_score = 0.0
        
        # Intent-based relevance
        if primary_intent == 'measurement' and any(word in content for word in ['dimension', 'size', 'feet', 'inches']):
            is_relevant = True
            relevance_score = 0.9
        elif primary_intent == 'count' and 'drawing_area' in meta.get('label', ''):
            is_relevant = True 
            relevance_score = 0.8
        elif primary_intent == 'annotation' and any(word in content for word in ['note', 'label', 'text']):
            is_relevant = True
            relevance_score = 0.7
        elif primary_intent == 'material' and any(word in content for word in ['material', 'finish', 'construction']):
            is_relevant = True
            relevance_score = 0.8
        
        # Keyword-based relevance
        query_words = query_lower.split()
        content_matches = sum(1 for word in query_words if word in content)
        if content_matches > 0:
            is_relevant = True
            relevance_score = max(relevance_score, content_matches / len(query_words))
        
        # Extract bounding box if element is relevant and has spatial data
        if is_relevant and 'bbox' in meta:
            try:
                bbox = meta['bbox']
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    relevant_boxes.append({
                        'bbox': bbox,  # [x1, y1, x2, y2]
                        'content': element['content'][:100],  # Truncate for display
                        'element_type': meta.get('label', 'unknown'),
                        'source': element['source'],
                        'relevance_score': relevance_score,
                        'sheet_id': meta.get('sheet_id', '')
                    })
            except:
                continue
    
    # Sort by relevance score and return top matches
    relevant_boxes.sort(key=lambda x: x['relevance_score'], reverse=True)
    return relevant_boxes[:10]  # Limit to top 10 for performance

def format_measurements_with_spatial_context(measurements: List[Dict]) -> str:
    """Format measurements with their spatial drawing context"""
    if not measurements:
        return "No measurements found."
    
    # Group measurements by drawing region/element type
    by_region = {}
    for m in measurements:
        region = m.get('drawing_region', 'unknown')
        if region not in by_region:
            by_region[region] = []
        by_region[region].append(m)
    
    formatted_parts = []
    for region, region_measurements in by_region.items():
        formatted_parts.append(f"\n📐 **{region.upper()} MEASUREMENTS:**")
        
        for m in region_measurements[:5]:  # Limit to 5 per region
            context = m.get('content_context', '').strip()
            bbox = m.get('spatial_data')
            
            measurement_info = f"  • {m['measurement']}"
            if context:
                measurement_info += f" - Context: \"{context}...\""
            if bbox and len(bbox) >= 4:
                measurement_info += f" - Location: [{bbox[0]:.0f},{bbox[1]:.0f}] to [{bbox[2]:.0f},{bbox[3]:.0f}]"
            
            formatted_parts.append(measurement_info)
    
    return "\n".join(formatted_parts)

def analyze_pre_extracted_dimension_region(dimension_image_path: str, sheet_id: str, region_content: str) -> str:
    """Analyze pre-extracted dimension line image with GPT-4o Vision"""
    
    try:
        import base64
        import os
        
        # Convert the original path to our local path structure
        # Original: /content/drive/MyDrive/main-drawings-embeddings/extracted_regions/...
        # Local: /home/vikramcode/fresco/annotation_images/annotation_images/...
        
        original_filename = os.path.basename(dimension_image_path)
        local_path = f"/home/vikramcode/fresco/annotation_images/annotation_images/{original_filename}"
        
        # If the direct mapping doesn't work, try to find it in the annotation_images directory
        if not os.path.exists(local_path):
            annotation_dir = "/home/vikramcode/fresco/annotation_images/annotation_images"
            if os.path.exists(annotation_dir):
                # Look for files containing the sheet ID and dimension_line
                for file in os.listdir(annotation_dir):
                    if sheet_id.split('_')[0] in file and 'dimension' in file.lower():
                        local_path = os.path.join(annotation_dir, file)
                        break
        
        if not os.path.exists(local_path):
            return f"Dimension image not found at {local_path}"
        
        # Load and encode the pre-extracted dimension image
        with open(local_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Analyze the pre-extracted dimension region
        dimension_prompt = f"""You are an expert architectural analyst examining a dimension line region from sheet {sheet_id}.

This image shows a pre-extracted dimension line region from an architectural drawing. Please analyze and extract:

1. **EXACT MEASUREMENTS**: What specific numerical values do you see? (include units: feet, inches, mm)
2. **WHAT'S BEING MEASURED**: What building elements are being dimensioned?
   - Room dimensions (length, width, height)
   - Wall thicknesses  
   - Door/window sizes
   - Equipment clearances
   - Structural spacing
3. **DIMENSION LABELS**: Any text labels, room names, or callouts near the dimensions
4. **DIMENSION LINES**: Describe the dimension line arrows, extension lines, and leaders
5. **SPATIAL CONTEXT**: What part of the building this dimension relates to
6. **TEXT ANNOTATIONS**: Any text or callouts that explain what's being measured

Text context: "{region_content[:100]}..."

Be very specific about the measurements you can see and what they're measuring. This is a focused dimension line region."""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": dimension_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=400
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Vision analysis error for dimension region: {e}"

def analyze_dimension_lines_with_vision(document_data: Dict, sheet_id: str) -> Dict[str, str]:
    """Automatically run vision analysis on dimension line regions to extract measurements and labels"""
    
    dimension_analyses = {}
    
    try:
        # Find all dimension line elements
        image_elements = document_data['image_elements']
        dimension_regions = []
        
        for doc, meta in zip(image_elements['documents'], image_elements['metadatas']):
            if meta.get('label', '').lower() in ['dimension_line', 'dimension', 'dim_line']:
                # Try different bbox keys
                bbox = meta.get('bbox') or meta.get('bbox_norm') or meta.get('spatial', {}).get('bbox')
                dimension_regions.append({
                    'content': doc,
                    'metadata': meta,
                    'bbox': bbox
                })
        
        print(f"🔍 Found {len(dimension_regions)} dimension line regions in {sheet_id}")
        
        # Analyze each dimension region with vision (limit to top 3 for cost control)
        for i, region in enumerate(dimension_regions[:3]):
            try:
                bbox = region['bbox']
                if not bbox or len(bbox) < 4:
                    continue
                
                print(f"🖼️ Running vision analysis on dimension region {i+1} at {bbox}")
                
                # Crop and analyze the dimension region with GPT-4o Vision
                vision_analysis = crop_and_analyze_dimension_region(
                    sheet_id, 
                    bbox, 
                    region['content']
                )
                
                dimension_analyses[f"region_{i}"] = {
                    'bbox': bbox,
                    'vision_analysis': vision_analysis,
                    'content_context': region['content'][:100],
                    'region_location': f"[{bbox[0]},{bbox[1]}] to [{bbox[2]},{bbox[3]}]"
                }
                
            except Exception as e:
                print(f"Error analyzing dimension region {i}: {e}")
                continue
        
        return dimension_analyses
        
    except Exception as e:
        print(f"Error in dimension line vision analysis: {e}")
        return {}

def analyze_document_text_only(sheet_id: str, query: str, intent_info: Dict) -> Dict[str, Any]:
    """Analyze document using only text embeddings - no vision analysis needed"""
    print(f"📝 Text-only analysis for {sheet_id} - Intent: {intent_info['primary_intent']}")
    
    try:
        document_data = get_all_document_elements(sheet_id)
        if "error" in document_data:
            return {'error': document_data['error']}
        
        # Focus on text embeddings with intent-based filtering
        text_docs = document_data['text_elements']['documents']
        text_metas = document_data['text_elements']['metadatas']
        
        # Filter and organize based on query intent
        primary_intent = intent_info['primary_intent']
        relevant_content = []
        measurements = []
        
        # Separate dimension_line elements for special processing
        dimension_elements = []
        
        for doc, meta in zip(text_docs, text_metas):
            element_type = meta.get('element_type', 'text')
            label = meta.get('label', '')
            
            # Collect dimension_line elements separately
            if label == 'dimension_line' or 'dimension' in element_type.lower():
                dimension_elements.append((doc, meta))
                continue
            
            # Intent-based content filtering for non-dimension elements
            if primary_intent == 'annotation' and 'note' in element_type.lower():
                relevant_content.append(f"📝 {element_type}: {doc}")
            elif primary_intent == 'material' and any(word in doc.lower() for word in ['material', 'finish', 'construction', 'specification']):
                relevant_content.append(f"🏗️ {element_type}: {doc}")
            elif primary_intent == 'location' and any(word in doc.lower() for word in ['room', 'space', 'area', 'located']):
                relevant_content.append(f"📍 {element_type}: {doc}")
            elif primary_intent == 'description':
                relevant_content.append(f"📋 {element_type}: {doc}")
            else:
                # Include all content for general queries
                relevant_content.append(f"• {element_type}: {doc}")
        
        # Process dimension_line elements with LLM interpretation
        dimension_analysis = None
        if dimension_elements and (primary_intent in ['measurement', 'count'] or any(word in query.lower() for word in ['dimension', 'size', 'height', 'width', 'room', 'big', 'large', 'area'])):
            print(f"📐 Found {len(dimension_elements)} dimension_line elements, using LLM interpretation")
            
            # Search for relevant dimension texts
            relevant_dimensions = search_dimension_texts(dimension_elements, query)
            
            if relevant_dimensions:
                # Use LLM to interpret dimension texts
                dimension_analysis = interpret_dimensions_with_llm(relevant_dimensions, query, sheet_id)
                
                # Add interpreted dimensions to measurements
                for dim in relevant_dimensions:
                    spatial_context = {
                        'measurement': dim['text'],  # Use full text instead of regex extract
                        'element_type': 'dimension_line',
                        'sheet_id': sheet_id,
                        'content_context': dim['text'],
                        'spatial_data': dim['bbox'],
                        'drawing_region': dim.get('element_type', 'dimension_line'),
                        'confidence': dim['relevance_score']
                    }
                    measurements.append(spatial_context)
        
        # Create focused context based on intent
        context_parts = [
            f"🎯 **TEXT-BASED ANALYSIS** - {sheet_id}",
            f"Intent: {primary_intent} (Vision not required)",
            f"Total Text Elements: {len(text_docs)}",
            f"Relevant Content:"
        ]
        
        # Limit relevant content to avoid token limits
        context_parts.extend(relevant_content[:20])
        
        # Add dimension analysis if available
        dimension_context = ""
        if dimension_analysis:
            dimension_parts = ["\n📐 **DIMENSION ANALYSIS** (LLM-interpreted):"]
            dimension_parts.append(f"Found {dimension_analysis['dimension_count']} dimension_line elements")
            dimension_parts.append(f"**Interpretation**: {dimension_analysis['interpretation']}")
            dimension_context = "\n".join(dimension_parts)
        
        # Format measurements with spatial context
        if measurements:
            measurements_formatted = format_measurements_with_spatial_context(measurements)
            context_parts.append(measurements_formatted)
        
        if dimension_context:
            context_parts.append(dimension_context)
        
        context = "\n".join(context_parts)
        
        # Use simplified prompt for text-only analysis
        prompt = f"""Answer this query about architectural drawing {sheet_id}:

QUERY: {query}

INFORMATION FROM DRAWING:
{context}

Answer the query directly using the information provided."""

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model for text-only
            messages=[
                {"role": "system", "content": "You are an architectural consultant providing text-based document analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.2
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract measurement values for the API response, prioritizing dimension analysis
        measurement_values = []
        if dimension_analysis and dimension_analysis['measurements']:
            measurement_values.extend(dimension_analysis['measurements'])
        
        if measurements:
            for m in measurements:
                if isinstance(m, dict):
                    # Return raw text from dimension_line elements, let LLM interpret
                    measurement_values.append(m['measurement'])  # Raw dimension text
                else:
                    measurement_values.append(str(m))
        
        return {
            'answer': answer,
            'analysis_type': 'text_only',
            'intent': primary_intent,
            'confidence': 0.8,  # Good confidence for text-based queries
            'context_summary': f"Text analysis of {sheet_id} - {len(relevant_content)} relevant items",
            'source_sheets': [sheet_id],
            'measurements': measurement_values[:10] if measurement_values else None,
        }
        
    except Exception as e:
        return {'error': f"Text-only analysis failed: {e}"}

def analyze_cross_documents(sheet_ids: List[str], query: str, max_docs: int = 2) -> Dict[str, Any]:
    """Analyze multiple documents for cross-document queries with cost optimization"""
    
    if len(sheet_ids) > max_docs:
        return {
            'error': f'Cross-document analysis limited to {max_docs} documents. Found {len(sheet_ids)} sheets: {sheet_ids}. Please analyze these separately or choose up to {max_docs} documents.',
            'suggested_queries': [f'Analyze document {sheet}' for sheet in sheet_ids[:max_docs]]
        }
    
    print(f"🔄 Cross-document analysis: {sheet_ids}")
    
    try:
        documents_data = {}
        vision_analyses = {}
        total_cost_estimate = 0
        
        # Get data for each document
        for sheet_id in sheet_ids:
            doc_data = get_all_document_elements(sheet_id)
            if "error" in doc_data:
                return {'error': f"Document {sheet_id} not found: {doc_data['error']}"}
            documents_data[sheet_id] = doc_data
            
            # Get vision analysis for document-specific parts of query
            try:
                vision_analysis = analyze_document_with_gpt4_vision(sheet_id, query)
                vision_analyses[sheet_id] = vision_analysis
            except Exception as e:
                vision_analyses[sheet_id] = f"Vision analysis failed: {e}"
        
        # Combine text embeddings from all documents  
        combined_text_context = []
        combined_measurements = []
        all_source_sheets = []
        
        for sheet_id, doc_data in documents_data.items():
            # Add text content with sheet identification
            text_docs = doc_data['text_elements']['documents']
            text_metas = doc_data['text_elements']['metadatas']
            
            combined_text_context.append(f"\n=== DOCUMENT {sheet_id} TEXT CONTENT ===")
            for i, (doc, meta) in enumerate(zip(text_docs, text_metas)):
                if i >= 10:  # Limit to 10 per doc
                    break
                element_type = meta.get('element_type', 'text')
                combined_text_context.append(f"[{sheet_id}] {element_type}: {doc}")
            
            # Extract measurements with spatial linking
            for doc, meta in zip(text_docs, text_metas):
                found_measurements = re.findall(r'\d+(?:\.\d+)?["\'-]|\d+(?:\.\d+)?\s*(?:ft|in|mm|cm|m)\b', str(doc))
                for measurement in found_measurements:
                    spatial_context = {
                        'measurement': measurement,
                        'sheet_id': sheet_id,
                        'element_type': meta.get('element_type', 'text'),
                        'drawing_region': meta.get('label', 'unknown'),
                        'context': doc[:80],  # Context snippet
                        'bbox': meta.get('bbox', None)
                    }
                    combined_measurements.append(spatial_context)
            
            all_source_sheets.append(sheet_id)
        
        # Create comprehensive cross-document prompt
        vision_context = "\n".join([f"=== {sheet_id} VISION ANALYSIS ===\n{analysis}" 
                                  for sheet_id, analysis in vision_analyses.items()])
        
        text_context = "\n".join(combined_text_context[:50])  # Limit context size
        
        cross_doc_prompt = f"""You are a senior architectural consultant performing cross-document analysis.

QUERY: {query}

DOCUMENTS ANALYZED: {', '.join(sheet_ids)}

=== VISION ANALYSIS FROM SOURCE DRAWINGS ===
{vision_context}

=== TEXT EMBEDDINGS CONTEXT ===  
{text_context}

Please provide a comprehensive comparative analysis that:
1. **COMPARES** the documents directly based on the query
2. **IDENTIFIES** similarities and differences between the sheets
3. **CROSS-REFERENCES** information from both vision analysis and text embeddings
4. **HIGHLIGHTS** relationships and dependencies between documents
5. **LINKS DIMENSIONS TO DRAWINGS**: When mentioning measurements, always specify:
   - Which specific drawing or drawing region the dimension belongs to
   - What building element or space the measurement relates to
   - How measurements from different sheets coordinate or conflict
6. **SPATIAL COORDINATION**: Explain how dimensions and elements in one sheet relate to those in the other sheet

Focus on architectural relationships, design consistency, technical coordination, and spatial relationships between the documents."""
        
        # Generate cross-document response with cost tracking
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert architectural consultant specializing in cross-document analysis and design coordination."},
                {"role": "user", "content": cross_doc_prompt}
            ],
            max_tokens=600,
            temperature=0.2
        )
        
        
        answer = response.choices[0].message.content.strip()
        
        # Format measurements with spatial context for response
        measurement_summary = []
        if combined_measurements:
            for m in combined_measurements[:15]:
                if isinstance(m, dict):
                    measurement_summary.append(f"{m['measurement']} - {m['sheet_id']} ({m.get('drawing_region', 'unknown')})")
                else:
                    measurement_summary.append(str(m))

        return {
            'answer': answer,
            'documents_analyzed': sheet_ids,
            'vision_analyses': vision_analyses,
            'confidence': 0.85,  # High confidence for cross-document analysis
            'context_summary': f"Cross-document analysis of {len(sheet_ids)} sheets with combined vision and text analysis",
            'source_sheets': all_source_sheets,
            'measurements': measurement_summary if measurement_summary else None,
            'analysis_type': 'cross_document'
        }
        
    except Exception as e:
        return {'error': f"Cross-document analysis failed: {e}"}

def extract_target_element(query: str, sheet_id: str) -> str:
    """Extract the target element/room/space from the query"""
    import re
    
    # Remove sheet ID references from query
    query_clean = query
    for pattern in [sheet_id, sheet_id.split('_')[0]]:
        query_clean = re.sub(rf'\b{re.escape(pattern)}\b', '', query_clean, flags=re.IGNORECASE)
    
    # Remove common query words
    stop_words = ['detailed', 'analysis', 'analyze', 'describe', 'what', 'is', 'the', 'in', 'of', 'for', 'about', 'tell', 'me']
    words = query_clean.strip().split()
    filtered_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
    
    # Join remaining words as the target element
    target_element = ' '.join(filtered_words).strip()
    
    # Handle common architectural terms
    if not target_element:
        return "main space"
    
    return target_element

def search_element_in_document(target_element: str, document_data: Dict) -> Dict:
    """Search for target element in document text embeddings and find associated drawing area"""
    
    try:
        text_docs = document_data['text_elements']['documents']
        text_metas = document_data['text_elements']['metadatas']
        
        best_match = None
        best_score = 0
        target_lower = target_element.lower()
        
        # Search through text embeddings for the target element
        for doc, meta in zip(text_docs, text_metas):
            doc_lower = doc.lower()
            
            # Calculate relevance score
            if target_lower in doc_lower:
                score = 1.0  # Exact match
            else:
                # Partial match scoring
                target_words = set(target_lower.split())
                doc_words = set(doc_lower.split())
                common_words = target_words.intersection(doc_words)
                score = len(common_words) / len(target_words) if target_words else 0
            
            if score > best_score:
                best_score = score
                best_match = {
                    'text_content': doc,
                    'metadata': meta,
                    'relevance_score': score
                }
        
        if best_match and best_score > 0.3:  # Minimum relevance threshold
            return {
                'found': True,
                'target_element': target_element,
                'matching_text': best_match['text_content'],
                'metadata': best_match['metadata'],
                'relevance_score': best_match['relevance_score'],
                'drawing_area_bbox': best_match['metadata'].get('bbox'),
                'element_type': best_match['metadata'].get('element_type', 'unknown')
            }
        else:
            return {
                'found': False,
                'target_element': target_element,
                'reason': 'No matching text embeddings found'
            }
            
    except Exception as e:
        return {
            'found': False,
            'target_element': target_element,
            'reason': f'Search error: {e}'
        }

def analyze_specific_drawing_area(sheet_id: str, target_element: str, bbox: list, original_query: str) -> str:
    """Analyze specific drawing area using vision with original query as prompt"""
    import chromadb
    import base64
    import os
    from PIL import Image
    
    try:
        # Get the source image
        client = chromadb.PersistentClient(path="./image_vectordb")
        collection = client.get_collection("source_images")
        
        all_results = collection.get(include=['metadatas'])
        target_image = None
        
        for meta in all_results['metadatas']:
            if sheet_id in meta['sheet_id'] or meta['sheet_id'] in sheet_id:
                target_image = meta
                break
        
        if not target_image:
            return f"❌ Source image not found for {sheet_id}"
        
        print(f"🎯 TARGETED ANALYSIS: {target_element} in {sheet_id} at bbox {bbox}")
        
        # Load and optionally crop the image if bbox is available
        with open(target_image['image_path'], "rb") as image_file:
            if bbox and len(bbox) >= 4:
                # Crop to the specific region
                image = Image.open(target_image['image_path'])
                x1, y1, x2, y2 = bbox
                cropped = image.crop((x1, y1, x2, y2))
                
                # Convert cropped image to base64
                import io
                buffer = io.BytesIO()
                cropped.save(buffer, format='PNG')
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                analysis_type = "cropped region"
            else:
                # Use full image
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                analysis_type = "full sheet"
        
        # Use original query directly as the vision prompt
        vision_prompt = f"""You are an expert architectural analyst examining a drawing from sheet {sheet_id}.

SPECIFIC FOCUS: {target_element}

ORIGINAL QUERY: {original_query}

You are looking at {analysis_type} that should contain information about "{target_element}". Please provide a detailed analysis focusing specifically on this element. Examine what you can see and provide comprehensive information about:

- The specific architectural features of {target_element}
- Dimensions, measurements, and specifications visible
- Materials and finishes shown for this element
- Spatial relationships and connections to other elements
- Construction details and technical annotations
- Any text, labels, or callouts related to {target_element}

Focus your analysis specifically on "{target_element}" and answer the user's original query about this element."""

        # Analyze with GPT-4o Vision
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=600
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Targeted vision analysis error for {target_element} in {sheet_id}: {e}"

def analyze_cropped_element(crop_path: str, target_element: str, original_query: str) -> str:
    """Analyze a specific cropped element using vision"""
    import base64
    
    try:
        print(f"🎯 TARGETED ANALYSIS: Analyzing {target_element} from crop: {crop_path}")
        
        # Load the cropped image
        with open(crop_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Use original query directly as the vision prompt
        vision_prompt = f"""You are an expert architectural analyst examining a specific element from a construction drawing.

SPECIFIC FOCUS: {target_element}

ORIGINAL QUERY: {original_query}

You are looking at a cropped region that contains "{target_element}". Please provide a detailed analysis focusing specifically on this element. Examine what you can see and provide comprehensive information about:

- The specific architectural features of {target_element}
- Dimensions, measurements, and specifications visible
- Materials and finishes shown for this element
- Spatial relationships and connections to other elements
- Construction details and technical annotations
- Any text, labels, or callouts related to {target_element}

Focus your analysis specifically on "{target_element}" and answer the user's original query about this element."""

        # Analyze with GPT-4o Vision
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=600
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Cropped element analysis error for {target_element}: {e}"

def analyze_document_detailed_vision(sheet_id: str, query: str) -> str:
    """Detailed vision analysis with targeted element extraction"""
    import chromadb
    import base64
    import os
    
    try:
        # Try targeted element analysis first
        target_element = extract_target_element(query, sheet_id)
        print(f"🎯 Extracted target element: '{target_element}'")
        
        # Use existing RAG system to search for the target element
        if rag_system:
            search_query = f"{target_element} in {sheet_id}"
            rag_results = rag_system.ask(search_query)
            
            # Look for visual elements with crop paths
            for result in rag_results.get('results', []):
                metadata = result.get('metadata', {})
                if metadata.get('crop_path') and sheet_id in metadata.get('sheet_id', ''):
                    print(f"✅ Found '{target_element}' with visual crop, using targeted analysis")
                    return analyze_cropped_element(metadata['crop_path'], target_element, query)
            
            print(f"⚠️ '{target_element}' not found with visual crop, falling back to full sheet analysis")
        else:
            print(f"⚠️ RAG system not available, falling back to full sheet analysis")
        
        # Fallback to full sheet analysis
        # Get image from our dedicated image vector DB
        client = chromadb.PersistentClient(path="./image_vectordb")
        collection = client.get_collection("source_images")
        
        # Find the specific sheet image
        all_results = collection.get(include=['metadatas'])
        target_image = None
        
        for meta in all_results['metadatas']:
            if sheet_id in meta['sheet_id'] or meta['sheet_id'] in sheet_id:
                target_image = meta
                break
        
        if not target_image:
            return f"❌ Source image not found for {sheet_id}"
        
        print(f"🔍 DETAILED ANALYSIS: Loading source image: {target_image['filename']}")
        
        # Load and encode the complete source image
        with open(target_image['image_path'], "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Use the original query directly as the vision prompt
        vision_prompt = f"""You are an expert architectural analyst examining this complete construction drawing sheet: {sheet_id}

ORIGINAL QUERY: {query}

Please provide a comprehensive detailed analysis of this architectural drawing sheet, focusing specifically on what the user is asking about. Examine the drawing carefully and provide detailed information about:

- All visible architectural elements and spaces relevant to the query
- Construction details and specifications shown
- Materials and finishes visible in the drawing
- Dimensions and measurements you can read
- Room/space identifications and their relationships
- Technical annotations, notes, and callouts
- Drawing organization, views, and layout
- Any specific elements mentioned in the user's query

Give a thorough visual analysis based on what you can actually see in this architectural drawing."""

        # Analyze with GPT-4o Vision
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Detailed vision analysis error for {sheet_id}: {e}"

def analyze_document_with_gpt4_vision(sheet_id: str, query: str) -> str:
    """Analyze complete document image with GPT-4o Vision - only for document-specific queries"""
    import chromadb
    import base64
    import os
    
    try:
        # Get image from our dedicated image vector DB
        client = chromadb.PersistentClient(path="./image_vectordb")
        collection = client.get_collection("source_images")
        
        # Find the specific sheet image
        all_results = collection.get(include=['metadatas'])
        target_image = None
        
        for meta in all_results['metadatas']:
            if sheet_id in meta['sheet_id'] or meta['sheet_id'] in sheet_id:
                target_image = meta
                break
        
        if not target_image:
            return f"❌ Source image not found for {sheet_id}"
        
        print(f"🖼️ Loading source image: {target_image['filename']}")
        
        # Load and encode the complete source image
        with open(target_image['image_path'], "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create focused prompt based on query intent
        if any(word in query.lower() for word in ['how many', 'count', 'drawings']):
            vision_prompt = f"""You are an expert architectural analyst examining this complete construction drawing sheet: {sheet_id}

QUERY: {query}

Please analyze this architectural drawing sheet and provide:

1. **EXACT COUNT**: How many distinct drawings, elevations, sections, or views are shown on this sheet?
2. **DRAWING TYPES**: What type of architectural drawings are these (elevations, sections, plans, details)?
3. **CONTENT DESCRIPTION**: What architectural spaces, rooms, or elements are depicted?
4. **ORGANIZATION**: How are the drawings arranged and labeled on the sheet?
5. **TECHNICAL DETAILS**: Any visible dimensions, scales, materials, or annotations?

Be very specific about what you can actually see in this complete architectural sheet. Count carefully - multiple views of the same space should be counted as separate drawings if they show different elevations or perspectives."""

        else:
            vision_prompt = f"""You are an expert architectural analyst examining this construction drawing sheet: {sheet_id}

QUERY: {query}

Please provide a comprehensive analysis of this architectural drawing including:
- All visible architectural elements and spaces
- Construction details and specifications
- Materials and finishes mentioned
- Dimensions and measurements shown
- Room/space identifications
- Any technical annotations or notes
- Overall drawing organization and content

Focus on answering the specific query while providing rich architectural context."""

        # Analyze with GPT-4o Vision
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=600
        )
        
        return {
            'analysis': response.choices[0].message.content.strip(),
            'image_data': image_data,
            'image_filename': target_image['filename']
        }
        
    except Exception as e:
        return {
            'analysis': f"Vision analysis error for {sheet_id}: {e}",
            'image_data': None,
            'image_filename': None
        }

def analyze_actual_images_with_gpt4_vision(image_regions: List[Dict], query: str, max_images: int = 5) -> List[str]:
    """Analyze actual image files using GPT-4 Vision"""
    import base64
    import os
    from PIL import Image
    
    image_analyses = []
    images_processed = 0
    
    for region in image_regions[:max_images]:  # Limit to avoid token limits
        try:
            # Get image path from metadata 
            image_path = region['metadata'].get('image_path', '')
            if not image_path:
                continue
                
            # Convert to local path
            # Original: /content/drive/MyDrive/main-drawings-embeddings/extracted_regions/...
            # Local: /home/vikramcode/fresco/annotation_images/annotation_images/...
            filename = os.path.basename(image_path)
            local_path = f"/home/vikramcode/fresco/annotation_images/annotation_images/{filename}"
            
            if not os.path.exists(local_path):
                # Try without filename extension matching
                sheet_folder = image_path.split('/')[-3] if '/' in image_path else ''
                region_type = image_path.split('/')[-2] if '/' in image_path else ''
                
                # Look for similar files
                annotation_dir = "/home/vikramcode/fresco/annotation_images/annotation_images"
                for file in os.listdir(annotation_dir):
                    if sheet_folder in file and region_type in file:
                        local_path = os.path.join(annotation_dir, file)
                        break
            
            if not os.path.exists(local_path):
                continue
                
            # Load and encode image
            with open(local_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Analyze with GPT-4 Vision
            prompt = f"""Analyze this architectural drawing region. This is from sheet {region['metadata'].get('sheet_id', 'unknown')} and represents a {region['metadata'].get('label', 'unknown')} element.

Query context: {query}

Please identify:
1. What architectural elements are shown (elevations, sections, plans, details)
2. How many distinct drawings or views are visible
3. Any text, labels, or dimensions visible
4. Spatial relationships and layout

Be specific about what you can see in this image region."""

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            analysis = response.choices[0].message.content
            image_analyses.append(f"Region {images_processed + 1} ({region['metadata'].get('label', 'unknown')}): {analysis}")
            images_processed += 1
            
        except Exception as e:
            print(f"Error analyzing image {region['metadata'].get('image_path', 'unknown')}: {e}")
            continue
    
    return image_analyses

def analyze_document_with_vision_aggregation(sheet_id: str, document_data: Dict, query: str) -> str:
    """Advanced document analysis with vision aggregation for all regions"""
    
    try:
        print(f"🎨 Starting vision aggregation analysis for {sheet_id}")
        
        # Get ALL regions (text + image) from the document
        text_elements = document_data['text_elements']
        image_elements = document_data['image_elements'] 
        
        # Categorize elements by type
        regions_by_type = {}
        all_regions = []
        
        # Process image elements (drawing areas, dimensions, etc.)
        for doc, meta in zip(image_elements['documents'], image_elements['metadatas']):
            element_type = meta.get('label', 'unknown')
            if element_type not in regions_by_type:
                regions_by_type[element_type] = []
            
            region_info = {
                'content': doc,
                'type': element_type,
                'metadata': meta,
                'source': 'image'
            }
            regions_by_type[element_type].append(region_info)
            all_regions.append(region_info)
        
        # Process text elements 
        for doc, meta in zip(text_elements['documents'], text_elements['metadatas']):
            element_type = meta.get('label', 'text')
            if element_type not in regions_by_type:
                regions_by_type[element_type] = []
                
            region_info = {
                'content': doc,
                'type': element_type, 
                'metadata': meta,
                'source': 'text'
            }
            regions_by_type[element_type].append(region_info)
            all_regions.append(region_info)
        
        # Focus analysis based on query intent
        if any(word in query.lower() for word in ['how many', 'count', 'drawings', 'elevations']):
            focus_types = ['drawing_area', 'title_block']
        elif any(word in query.lower() for word in ['dimension', 'measurement', 'size']):
            focus_types = ['dimension_line', 'drawing_area']
        else:
            focus_types = list(regions_by_type.keys())
        
        # Aggregate analysis
        analysis_parts = [f"COMPREHENSIVE VISION ANALYSIS - {sheet_id}"]
        analysis_parts.append(f"Total Regions Analyzed: {len(all_regions)}")
        analysis_parts.append(f"Region Types: {list(regions_by_type.keys())}")
        
        for element_type in focus_types:
            if element_type in regions_by_type:
                regions = regions_by_type[element_type]
                analysis_parts.append(f"\n{element_type.upper()} ANALYSIS ({len(regions)} regions):")
                
                for i, region in enumerate(regions):
                    analysis_parts.append(f"  Region {i+1}: {region['content'][:100]}...")
                    if 'spatial' in region['metadata']:
                        analysis_parts.append(f"    Spatial: Has bounding box and location data")
        
        # ACTUAL IMAGE ANALYSIS with GPT-4 Vision
        image_regions = [region for region in all_regions if region['source'] == 'image' and region['type'] == 'drawing_area']
        
        if image_regions:
            print(f"🖼️ Analyzing {len(image_regions)} actual images with GPT-4 Vision")
            actual_image_analyses = analyze_actual_images_with_gpt4_vision(image_regions, query, max_images=3)
            
            if actual_image_analyses:
                analysis_parts.append(f"\n🎯 ACTUAL IMAGE ANALYSIS RESULTS:")
                for analysis in actual_image_analyses:
                    analysis_parts.append(f"  {analysis}")
        
        # Special analysis for drawing areas (most important for counting)
        if 'drawing_area' in regions_by_type:
            drawing_areas = regions_by_type['drawing_area']
            analysis_parts.append(f"\nDRAWING COUNT ANALYSIS:")
            analysis_parts.append(f"Found {len(drawing_areas)} drawing area regions")
            
            # Try to determine if these are distinct drawings or parts of same drawing
            distinct_drawings = []
            for region in drawing_areas:
                content = region['content'].lower()
                # Look for indicators of separate drawings
                if any(indicator in content for indicator in ['elevation', 'section', 'detail', 'plan', 'view']):
                    distinct_drawings.append(region)
            
            analysis_parts.append(f"Likely distinct drawings: {len(distinct_drawings)}")
        
        context_for_gpt = "\n".join(analysis_parts)
        
        # Use GPT-4 for intelligent aggregation with emphasis on actual vision analysis
        prompt = f"""You are a senior architectural analyst with expertise in technical drawing interpretation. You have access to ACTUAL IMAGE ANALYSIS from GPT-4 Vision.

DOCUMENT: {sheet_id}
QUERY: {query}

COMPREHENSIVE ANALYSIS INCLUDING ACTUAL IMAGES:
{context_for_gpt}

IMPORTANT: The "ACTUAL IMAGE ANALYSIS RESULTS" section above contains real visual analysis of the drawing images using computer vision. This is the most accurate source for counting drawings and identifying content.

Based on this detailed analysis combining vision and metadata, provide a precise answer that:

1. **PRIORITIZES VISION ANALYSIS**: Give highest weight to the actual image analysis results when counting drawings
2. **COUNTS accurately**: For "how many drawings" questions, rely primarily on what GPT-4 Vision actually saw in the images
3. **IDENTIFIES precisely**: What types of architectural drawings are shown based on visual evidence
4. **DESCRIBES content**: What architectural elements are actually visible in the images
5. **CROSS-REFERENCES**: Compare vision results with metadata to ensure accuracy

Answer the query with confidence based on the actual visual analysis of {len(image_regions) if 'image_regions' in locals() else 0} drawing regions."""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert architectural document analyst. Provide precise, technical answers based on comprehensive region analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Vision aggregation analysis error: {e}"

def generate_response_with_openai(query: str, context: str, parsed_query: Dict) -> str:
    """Generate response using OpenAI"""
    prompt = f"""Answer this query based on the architectural drawing information provided:

QUERY: {query}

DRAWING INFORMATION:
{context[:2000]}

Answer the query directly using the information provided."""

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an architectural consultant providing precise technical answers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
    except:
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {"role": "system", "content": "You are an architectural consultant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Based on the available context, I can provide information about your architectural query. Error: {e}"

@app.get("/")
async def root():
    return {"message": "Fresco Architectural RAG API", "status": "online"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "rag_system": "available" if rag_system else "unavailable",
        "openai_configured": bool(openai.api_key),
        "system_type": "intelligent" if HAS_INTELLIGENT else ("basic" if HAS_BASIC_RAG else "none")
    }

@app.get("/inventory")
async def get_inventory():
    """Get document inventory"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        if HAS_INTELLIGENT:
            # Use intelligent system inventory
            inventory = rag_system.rag_system.get_drawing_inventory()
        else:
            # Use basic system inventory
            inventory = rag_system.get_drawing_inventory()
        
        return inventory
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inventory error: {e}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process architectural query"""
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        # Detect document query patterns and classify query type
        document_query_info = detect_document_query(request.query)
        
        # Classify query intent to determine if vision analysis is needed
        intent_info = classify_query_intent(request.query)
        print(f"🧠 Query intent analysis: {intent_info['primary_intent']} (Vision required: {intent_info['requires_vision']})")
        
        # Parse query
        parsed_query = parse_query_with_openai(request.query)
        
        # Handle multiple document queries as filtered search
        if document_query_info['sheets'] and len(document_query_info['sheets']) > 1:
            detected_sheets = document_query_info['sheets']
            
            # If no sheets detected but cross-document keywords found, try to extract from query
            if not detected_sheets and document_query_info['cross_document_keywords']:
                print(f"🔍 Multiple document keywords detected, attempting to extract sheet names from: '{request.query}'")
                import re
                manual_matches = re.findall(r'[A-Z][\d\-\.]+(?:\.\d+)?', request.query.upper())
                detected_sheets = manual_matches
                print(f"Manually extracted sheets: {detected_sheets}")
            
            # Expand short sheet names to full names
            if detected_sheets:
                expanded_sheets = expand_sheet_names(detected_sheets)
                print(f"🔍 Multi-document filtered search: {detected_sheets} → {expanded_sheets}")
                
                # Use filtered RAG search limited to these specific sheets
                try:
                    if HAS_INTELLIGENT:
                        # Get normal search results then filter by sheet
                        result = rag_system.ask(request.query)
                        
                        # Filter source_sheets to only include the specified sheets
                        filtered_source_sheets = []
                        for sheet in result.get('source_sheets', []):
                            if any(target_sheet in sheet for target_sheet in expanded_sheets):
                                filtered_source_sheets.append(sheet)
                        
                        # Flatten measurements if they're nested lists - just return raw text
                        measurements = result.get('measurements', [])
                        if measurements and isinstance(measurements[0], list):
                            flattened = []
                            for item in measurements:
                                if isinstance(item, list):
                                    flattened.extend(item)
                                else:
                                    flattened.append(str(item))
                            result['measurements'] = flattened[:10]
                        elif measurements:
                            # Ensure all measurements are strings
                            result['measurements'] = [str(m) for m in measurements[:10]]
                        
                        # Update result with filtered sheets
                        result['source_sheets'] = filtered_source_sheets
                        result['context_summary'] = f"Filtered search across {len(expanded_sheets)} specified sheets: {', '.join(expanded_sheets[:3])}"
                        
                    else:
                        # Use basic RAG with manual filtering
                        search_results = rag_system.multimodal_search(
                            request.query, 
                            max_results=request.max_results * 2  # Get more results to filter
                        )
                        
                        # Process results manually for basic system - filter by specified sheets
                        context_parts = []
                        measurements = []
                        filtered_sheets = []
                        
                        for search_result in search_results:
                            # Only include results from specified sheets
                            result_sheet = search_result.get('sheet_id', '')
                            if any(target_sheet in result_sheet for target_sheet in expanded_sheets):
                                context_parts.append(f"Sheet {result_sheet}: {search_result.get('content', '')[:200]}")
                                filtered_sheets.append(result_sheet)
                                
                                # Extract measurements
                                content = search_result.get('content', '')
                                found_measurements = re.findall(r'\d+(?:\.\d+)?["\'-]|\d+(?:\.\d+)?\s*(?:ft|in|mm|cm|m)\b', content)
                                measurements.extend(found_measurements)
                        
                        context = "\n".join(context_parts)
                        answer = generate_response_with_openai(request.query, context, parsed_query)
                        
                        result = {
                            'answer': answer,
                            'confidence': 0.8,
                            'context_summary': f"Filtered search across {len(expanded_sheets)} specified sheets",
                            'source_sheets': list(set(filtered_sheets)),
                            'measurements': list(set(measurements))[:10] if measurements else None
                        }
                    
                    return QueryResponse(
                        answer=result['answer'],
                        confidence=result['confidence'],
                        context_summary=result['context_summary'],
                        source_sheets=result['source_sheets'],
                        measurements=result.get('measurements'),
                        parsed_query=parsed_query
                    )
                        
                except Exception as e:
                    print(f"❌ Error in multi-document filtered search: {e}")
                    raise HTTPException(status_code=500, detail=f"Multi-document search error: {e}")
            else:
                # No sheets found for multi-document search
                return QueryResponse(
                    answer="❌ Multiple documents mentioned but no document sheets could be identified. Please specify sheet IDs like 'Check A8.4, M-2.1, and C400'",
                    confidence=0.3,
                    context_summary="Multi-document query without identifiable sheets",
                    source_sheets=[],
                    parsed_query=parsed_query
                )
        
        # Handle single document-specific queries with intelligent routing
        elif document_query_info['is_single_document']:
            document_id = document_query_info['sheets'][0]
            print(f"🎯 Document-specific query detected: {document_id}")
            
            # Check for detailed analysis request first
            if any(phrase in request.query.lower() for phrase in ['detailed analysis', 'detailed analyze', 'comprehensive analysis']):
                print(f"🔍 DETAILED ANALYSIS requested for {document_id}")
                try:
                    detailed_vision_result = analyze_document_detailed_vision(document_id, request.query)
                    
                    return QueryResponse(
                        answer=detailed_vision_result,
                        confidence=0.95,
                        context_summary=f"Detailed vision analysis of {document_id}",
                        source_sheets=[document_id],
                        measurements=None,
                        parsed_query=parsed_query
                    )
                except Exception as e:
                    print(f"❌ Error in detailed analysis: {e}")
                    raise HTTPException(status_code=500, detail=f"Detailed analysis error: {e}")
            
            # Route based on whether vision analysis is required
            if not intent_info['requires_vision']:
                print(f"📝 Using text-only analysis for {intent_info['primary_intent']} query")
                try:
                    text_result = analyze_document_text_only(document_id, request.query, intent_info)
                    
                    if 'error' in text_result:
                        raise HTTPException(status_code=404, detail=text_result['error'])
                    
                    return QueryResponse(
                        answer=text_result['answer'],
                        confidence=text_result['confidence'],
                        context_summary=text_result['context_summary'],
                        source_sheets=text_result['source_sheets'],
                        measurements=text_result.get('measurements'),
                        parsed_query=parsed_query
                    )
                except Exception as e:
                    print(f"❌ Error in text-only processing: {e}")
                    import traceback
                    traceback.print_exc()
                    raise HTTPException(status_code=500, detail=f"Text analysis error: {e}")
            
            # Continue with vision analysis for queries that require it
            try:
                document_data = get_all_document_elements(document_id)
                print(f"✅ Got document data: {document_data.keys()}")
                
                if "error" in document_data:
                    print(f"❌ Document error: {document_data['error']}")
                    raise HTTPException(status_code=404, detail=document_data["error"])
                
                # Format ALL document content for OpenAI
                context_parts = []
                context_parts.append(f"COMPLETE DOCUMENT ANALYSIS FOR {document_data['sheet_id']}:")
                
                # Add sheet summary
                sheet_summary = document_data.get('sheet_summary', {})
                context_parts.append(f"Drawing Type: {sheet_summary.get('drawing_type', 'Unknown')}")
                context_parts.append(f"Total Elements: {sheet_summary.get('total_elements', 0)}")
                
                # Add ALL text content
                text_docs = document_data['text_elements']['documents']
                text_metas = document_data['text_elements']['metadatas'] 
                context_parts.append(f"\nTEXT CONTENT ({len(text_docs)} elements):")
                for i, (doc, meta) in enumerate(zip(text_docs, text_metas)):
                    element_type = meta.get('element_type', 'text')
                    context_parts.append(f"  {i+1}. [{element_type}] {doc}")
                
                # Add ALL image content
                image_docs = document_data['image_elements']['documents']
                image_metas = document_data['image_elements']['metadatas']
                context_parts.append(f"\nIMAGE CONTENT ({len(image_docs)} elements):")
                for i, (doc, meta) in enumerate(zip(image_docs, image_metas)):
                    element_type = meta.get('element_type', 'image')
                    context_parts.append(f"  {i+1}. [{element_type}] {doc}")
                
                # No need to extract measurements - text_elements contain all dimension_line content
            
                # Enhanced document-specific analysis combining vision + text embeddings  
                cost_estimate = 0
                enhanced_context_parts = []
                
                # ALWAYS get vision analysis for document-specific queries
                print(f"🎯 Getting GPT-4o Vision analysis + text embeddings")
                vision_result = None
                try:
                    vision_result = analyze_document_with_gpt4_vision(
                        document_data['sheet_id'], 
                        request.query
                    )
                    if isinstance(vision_result, dict):
                        enhanced_context_parts.append(f"🖼️ **VISION ANALYSIS**:\n{vision_result['analysis']}")
                    else:
                        # Handle old string return format
                        enhanced_context_parts.append(f"🖼️ **VISION ANALYSIS**:\n{vision_result}")
                        vision_result = {'analysis': vision_result, 'image_data': None, 'image_filename': None}
                except Exception as e:
                    enhanced_context_parts.append(f"⚠️ Vision analysis failed: {e}")
                    vision_result = {'analysis': f"Vision analysis failed: {e}", 'image_data': None, 'image_filename': None}
                
                # Skip dimension line vision analysis - using text-based interpretation instead
                print(f"📝 Using text-based dimension analysis (no vision needed)")
                
                # Add comprehensive text context with better organization
                enhanced_context_parts.append(f"\n📋 **TEXT EMBEDDINGS ANALYSIS**:")
                enhanced_context_parts.append(f"Document: {document_data['sheet_id']}")
                enhanced_context_parts.append(f"Total Elements: {document_data['total_elements']}")
                
                # Organize text content by type
                text_by_type = {}
                for doc, meta in zip(text_docs, text_metas):
                    element_type = meta.get('element_type', 'text')
                    if element_type not in text_by_type:
                        text_by_type[element_type] = []
                    text_by_type[element_type].append(doc)
                
                for element_type, docs in text_by_type.items():
                    enhanced_context_parts.append(f"\n**{element_type.upper()}** ({len(docs)} items):")
                    for i, doc in enumerate(docs[:5]):  # Limit per type
                        enhanced_context_parts.append(f"  • {doc}")
                
                # Add image content descriptions
                if image_docs:
                    enhanced_context_parts.append(f"\n🖼️ **IMAGE REGIONS** ({len(image_docs)} elements):")
                    for i, (doc, meta) in enumerate(zip(image_docs[:5], image_metas[:5])):
                        element_type = meta.get('element_type', 'image')
                        enhanced_context_parts.append(f"  • [{element_type}] {doc}")
                
                context = "\n".join(enhanced_context_parts)
                
                answer = generate_response_with_openai(request.query, context, parsed_query)
                
                enhanced_answer = answer
                
                return QueryResponse(
                    answer=enhanced_answer,
                    confidence=0.9,  # High confidence for document-specific queries
                    context_summary=f"Enhanced analysis of {document_data['sheet_id']} combining vision + {document_data['total_elements']} text elements",
                    source_sheets=[document_data['sheet_id']],
                    measurements=None,  # Let LLM handle dimension interpretation from text_elements
                    parsed_query=parsed_query,
                    image_data=vision_result.get('image_data') if vision_result else None,
                    image_filename=vision_result.get('image_filename') if vision_result else None
                )
            except Exception as e:
                print(f"❌ Error in document-specific processing: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Document processing error: {e}")
        
        # Handle inventory questions (only if not document-specific)
        if parsed_query.get('intent_type') == 'inventory' or any(word in request.query.lower() for word in ['how many', 'count', 'total']):
            if HAS_INTELLIGENT:
                inventory = rag_system.rag_system.get_drawing_inventory()
            else:
                inventory = rag_system.get_drawing_inventory()
            
            context = f"Document inventory: {inventory['total_sheets']} sheets, {inventory['total_elements']} elements. Drawing types: {list(inventory['drawing_types'].keys())}"
            
            answer = generate_response_with_openai(request.query, context, parsed_query)
            
            return QueryResponse(
                answer=answer,
                confidence=0.95,
                context_summary=f"Inventory: {inventory['total_sheets']} sheets, {inventory['total_elements']} elements",
                source_sheets=list(inventory.get('sheets_detail', {}).keys())[:5] if 'sheets_detail' in inventory else [],
                parsed_query=parsed_query
            )
        
        # Use RAG search
        if HAS_INTELLIGENT:
            # Use intelligent system for complex queries
            print(f"🔍 Calling rag_system.ask() with query: {request.query}")
            try:
                result = rag_system.ask(request.query)
                print(f"✅ Got result from intelligent system")
                
                # Fix context_summary if it's a dict
                context_summary = result['context_summary']
                if isinstance(context_summary, dict):
                    context_summary = f"Found {context_summary.get('drawings_found', 0)} drawings, {context_summary.get('dimensions_found', 0)} dimensions"
                
                # Flatten measurements if they're nested lists
                measurements = result.get('measurements', [])
                if measurements and isinstance(measurements[0], list):
                    # Flatten nested lists: [['8"', '6"'], ['4"', '1"']] -> ['8"', '6"', '4"', '1"']
                    flattened = []
                    for sublist in measurements:
                        if isinstance(sublist, list):
                            flattened.extend(sublist)
                        else:
                            flattened.append(sublist)
                    measurements = flattened[:10]  # Limit to 10 measurements
                
                # Enhance answer with follow-up suggestion for detailed document analysis
                source_sheets = result.get('source_sheets', [])[:5]
                enhanced_answer = result['answer']
                
                if source_sheets:
                    primary_sheet = source_sheets[0]
                    enhanced_answer += f"\n\n💡 **Want more details?** I found relevant information in sheet `{primary_sheet}`. Would you like me to analyze this specific document with detailed vision analysis? Just ask: \"Analyze document {primary_sheet}\" or \"How many drawings in {primary_sheet}?\""
                
                return QueryResponse(
                    answer=enhanced_answer,
                    confidence=result['confidence'],
                    context_summary=context_summary,
                    source_sheets=source_sheets,
                    measurements=measurements if measurements else None,
                    parsed_query=result.get('parsed_query')
                )
            except Exception as e:
                print(f"❌ Error in intelligent system ask(): {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Intelligent system error: {e}")
        else:
            # Use basic RAG system with OpenAI enhancement
            search_results = rag_system.multimodal_search(
                request.query, 
                max_results=request.max_results
            )
            
            # Format context for OpenAI
            context_parts = []
            source_sheets = []
            measurements = []
            
            for result in search_results[:3]:
                context_parts.append(f"Sheet {result.get('sheet_id', 'unknown')}: {result.get('content', '')[:200]}")
                if result.get('sheet_id'):
                    source_sheets.append(result['sheet_id'])
                    
                # Extract measurements
                content = result.get('content', '')
                found_measurements = re.findall(r'\d+(?:\.\d+)?["\'-]|\d+(?:\.\d+)?\s*(?:ft|in|mm|cm|m)\b', content)
                measurements.extend(found_measurements)
            
            context = "\n".join(context_parts)
            answer = generate_response_with_openai(request.query, context, parsed_query)
            
            # Add follow-up suggestion for basic RAG too
            unique_source_sheets = list(set(source_sheets))[:5]
            if unique_source_sheets:
                primary_sheet = unique_source_sheets[0]
                answer += f"\n\n💡 **Want more details?** I found relevant information in sheet `{primary_sheet}`. For detailed document analysis with vision, ask: \"Analyze document {primary_sheet}\" or \"How many drawings in {primary_sheet}?\""
            
            return QueryResponse(
                answer=answer,
                confidence=0.7,
                context_summary=f"Found {len(search_results)} relevant matches",
                source_sheets=unique_source_sheets,
                measurements=list(set(measurements))[:8] if measurements else None,
                parsed_query=parsed_query
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@app.post("/highlight")
async def get_drawing_highlights(request: QueryRequest):
    """Highlighting feature disabled - not currently working"""
    raise HTTPException(status_code=501, detail="Highlighting feature temporarily disabled")

@app.get("/drawing/{sheet_id}")
async def get_drawing_image(sheet_id: str):
    """Serve drawing images"""
    try:
        # Clean sheet_id and find corresponding image
        clean_sheet_id = sheet_id.replace('.png', '')
        
        # Try direct path first
        image_path = f"/home/vikramcode/fresco/annotation_images/annotation_images/{clean_sheet_id}.png"
        
        if os.path.exists(image_path):
            return FileResponse(image_path, media_type="image/png")
        else:
            # Try to expand short sheet name to full name
            expanded_sheets = expand_sheet_names([clean_sheet_id])
            if expanded_sheets and expanded_sheets[0] != clean_sheet_id:
                full_sheet_name = expanded_sheets[0]
                image_path = f"/home/vikramcode/fresco/annotation_images/annotation_images/{full_sheet_name}.png"
                
                if os.path.exists(image_path):
                    return FileResponse(image_path, media_type="image/png")
            
            raise HTTPException(status_code=404, detail=f"Drawing {clean_sheet_id} not found at {image_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving drawing: {e}")

@app.get("/frontend")
async def serve_frontend():
    """Serve the modern frontend"""
    return FileResponse("/home/vikramcode/fresco/modern_frontend.html", media_type="text/html")

@app.get("/test")
async def test_system():
    """Test system components"""
    
    results = {
        "rag_system": "available" if rag_system else "unavailable",
        "openai": "configured" if openai.api_key else "not configured",
        "system_type": "intelligent" if HAS_INTELLIGENT else ("basic" if HAS_BASIC_RAG else "none")
    }
    
    # Test inventory
    if rag_system:
        try:
            if HAS_INTELLIGENT:
                inventory = rag_system.rag_system.get_drawing_inventory()
            else:
                inventory = rag_system.get_drawing_inventory()
            results["inventory_test"] = f"✅ {inventory['total_sheets']} sheets"
        except Exception as e:
            results["inventory_test"] = f"❌ {e}"
    
    # Test OpenAI
    if openai.api_key:
        try:
            test_query = parse_query_with_openai("What are the beam dimensions?")
            results["openai_test"] = f"✅ {test_query.get('intent_type', 'unknown')}"
        except Exception as e:
            results["openai_test"] = f"❌ {e}"
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)