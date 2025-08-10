import os
import json
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
from arch_rag_system import ArchRAGSystem

# Try OpenAI first, then fallback options
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

@dataclass
class IntelligentContext:
    """Rich context that links all related information"""
    primary_content: str
    related_drawings: List[Dict[str, Any]]
    related_text_blocks: List[Dict[str, Any]]
    related_dimensions: List[Dict[str, Any]]
    related_schedules: List[Dict[str, Any]]
    title_context: List[Dict[str, Any]]
    spatial_relationships: Dict[str, Any]
    semantic_links: List[Dict[str, Any]]
    confidence_score: float

class IntelligentArchSystem:
    def __init__(self, vectordb_path="./vectordb_extracted/vectordb_bge_openclip-v2"):
        """
        Intelligent Architectural Analysis System
        
        Creates semantic and spatial links between ALL document elements:
        - Drawings ↔ Dimensions ↔ Text blocks ↔ Schedules ↔ Titles
        """
        print("🧠 Initializing Intelligent Architectural System...")
        
        # Initialize base RAG system
        self.rag_system = ArchRAGSystem(vectordb_path)
        
        # Initialize LLMs for intelligent processing
        self.query_parser_llm = self._initialize_query_parser()
        self.response_generator_llm = self._initialize_response_generator()
        
        # Build comprehensive knowledge graph
        print("🔗 Building knowledge graph...")
        self.knowledge_graph = self._build_knowledge_graph()
        
        print("✅ Intelligent Architectural System ready!")
    
    def _initialize_query_parser(self):
        """Initialize OpenAI for intelligent query parsing"""
        if not HAS_OPENAI:
            print("⚠️ OpenAI not available, using rule-based parser")
            return None
            
        try:
            # Try to get API key from environment or file
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                try:
                    with open('api_key', 'r') as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    print("⚠️ OPENAI_API_KEY not found in environment or api_key file")
                    return None
                
            print("🔄 Loading OpenAI query parser...")
            openai.api_key = api_key
            
            # Test the connection
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            
            print("✅ OpenAI query parser ready")
            return "openai"
            
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI: {e}")
            return None
    
    def _initialize_response_generator(self):
        """Initialize OpenAI for intelligent response generation"""
        if not HAS_OPENAI:
            return None
            
        try:
            # Use same API key setup as parser
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                try:
                    with open('api_key', 'r') as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    return None
                        
            print("🔄 Loading OpenAI response generator...")
            openai.api_key = api_key
            
            print("✅ OpenAI response generator ready")
            return "openai"
            
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI generator: {e}")
            return None
    
    def _build_knowledge_graph(self) -> Dict[str, Any]:
        """
        Build comprehensive knowledge graph linking all document elements
        
        This creates intelligent connections between:
        - Drawing areas ↔ Their dimension lines
        - Text blocks ↔ Related drawings
        - Schedules ↔ Referenced elements
        - Titles ↔ Document content
        """
        print("📊 Analyzing document relationships...")
        
        # Get all elements grouped by sheet
        inventory = self.rag_system.get_drawing_inventory()
        knowledge_graph = {
            'sheets': {},
            'global_relationships': {
                'measurement_to_drawing': {},
                'schedule_to_elements': {},
                'title_to_content': {},
                'semantic_clusters': {}
            },
            'statistics': inventory
        }
        
        # Process each sheet to build rich relationships
        sample_sheets = list(inventory['sheets_detail'].keys())[:10]  # Process subset for performance
        
        for sheet_id in sample_sheets:
            print(f"  🔍 Processing {sheet_id}...")
            sheet_knowledge = self._analyze_sheet_intelligence(sheet_id)
            knowledge_graph['sheets'][sheet_id] = sheet_knowledge
        
        return knowledge_graph
    
    def _analyze_sheet_intelligence(self, sheet_id: str) -> Dict[str, Any]:
        """
        Create intelligent analysis of a single sheet with ALL relationships
        """
        # Get all elements for this sheet
        sheet_data = self.rag_system.get_sheet_summary(sheet_id)
        
        # Organize elements by type for intelligent linking
        elements = sheet_data['detailed_elements']
        
        intelligence = {
            'sheet_metadata': {
                'id': sheet_id,
                'type': sheet_data['drawing_type'],
                'total_elements': sheet_data['total_elements']
            },
            'content_analysis': {},
            'intelligent_links': {
                'dimension_to_drawing': [],
                'text_to_drawing': [],
                'schedule_to_reference': [],
                'title_to_content': []
            },
            'extracted_knowledge': {
                'measurements': set(),
                'materials': set(),
                'specifications': set(),
                'room_names': set(),
                'equipment': set()
            }
        }
        
        # Process each element type intelligently
        for element_type, element_list in elements.items():
            intelligence['content_analysis'][element_type] = self._analyze_element_type(
                element_type, element_list, sheet_id
            )
        
        # Create intelligent cross-links
        intelligence['intelligent_links'] = self._create_intelligent_links(elements)
        
        # Extract domain knowledge
        intelligence['extracted_knowledge'] = self._extract_domain_knowledge(elements)
        
        return intelligence
    
    def _analyze_element_type(self, element_type: str, elements: List[Dict], sheet_id: str) -> Dict[str, Any]:
        """Intelligent analysis of specific element types"""
        analysis = {
            'count': len(elements),
            'avg_confidence': 0,
            'key_content': [],
            'spatial_distribution': [],
            'extracted_info': []
        }
        
        if not elements:
            return analysis
        
        # Calculate average OCR confidence
        confidences = [e.get('ocr_confidence', 0) for e in elements]
        analysis['avg_confidence'] = sum(confidences) / len(confidences)
        
        for element in elements:
            content = element.get('content', '')
            meta = element.get('metadata', {})
            spatial = element.get('spatial', {})
            
            # Extract key information based on element type
            if element_type == 'dimension_line':
                measurements = self.rag_system._extract_measurements(content)
                if measurements:
                    analysis['extracted_info'].append({
                        'measurements': measurements,
                        'context': content[:100],
                        'spatial': spatial,
                        'confidence': element.get('ocr_confidence', 0)
                    })
            
            elif element_type == 'text_block':
                # Extract structured information from text blocks
                extracted = self._extract_structured_info(content)
                if extracted:
                    analysis['extracted_info'].append({
                        'structured_info': extracted,
                        'full_text': content,
                        'spatial': spatial,
                        'confidence': element.get('ocr_confidence', 0)
                    })
            
            elif element_type == 'schedule_table':
                # Parse schedule/table content
                schedule_data = self._parse_schedule_content(content)
                analysis['extracted_info'].append({
                    'schedule_data': schedule_data,
                    'raw_content': content,
                    'spatial': spatial
                })
            
            elif element_type == 'title_block':
                # Extract title and sheet information
                title_info = self._parse_title_block(content)
                analysis['extracted_info'].append({
                    'title_info': title_info,
                    'raw_content': content
                })
            
            # Store spatial distribution
            if spatial and 'bbox_px' in spatial:
                analysis['spatial_distribution'].append(spatial['bbox_px'])
        
        return analysis
    
    def _extract_structured_info(self, text: str) -> Dict[str, List[str]]:
        """Extract structured information from text content"""
        if not text:
            return {}
        
        extracted = {
            'materials': [],
            'specifications': [],
            'room_names': [],
            'equipment': [],
            'codes_standards': [],
            'measurements': []
        }
        
        text_lower = text.lower()
        
        # Materials and grades
        material_patterns = [
            r'\b(?:steel|concrete|aluminum|wood|gypsum|plywood)\b',
            r'\bgrade\s*\d+\b',
            r'\bf[\'\u2032]c\s*=?\s*\d+\b',
            r'\bA\d{3}\b',  # ASTM standards
        ]
        
        for pattern in material_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['materials'].extend(matches)
        
        # Room and space names
        room_patterns = [
            r'\b(?:room|office|hall|lobby|corridor|mechanical|electrical|storage)\s+\w+\b',
            r'\b(?:classroom|laboratory|auditorium|library|cafeteria)\b',
        ]
        
        for pattern in room_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['room_names'].extend(matches)
        
        # Equipment and systems
        equipment_patterns = [
            r'\b(?:hvac|pump|fan|unit|system|panel|board)\b',
            r'\b(?:transformer|generator|switchgear|motor)\b',
        ]
        
        for pattern in equipment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['equipment'].extend(matches)
        
        # Codes and standards
        code_patterns = [
            r'\b(?:ASTM|ACI|AISC|UBC|IBC|NFPA)\s*\d*\b',
            r'\b(?:code|standard|specification|requirement)\b',
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['codes_standards'].extend(matches)
        
        # Measurements
        measurements = self.rag_system._extract_measurements(text)
        extracted['measurements'] = measurements
        
        # Remove empty lists and duplicates
        return {k: list(set(v)) for k, v in extracted.items() if v}
    
    def _parse_schedule_content(self, content: str) -> Dict[str, Any]:
        """Parse schedule/table content intelligently"""
        if not content:
            return {}
        
        # Look for table-like structures
        lines = content.split('\n')
        
        schedule_info = {
            'type': 'unknown',
            'entries': [],
            'headers': [],
            'key_values': {}
        }
        
        # Detect schedule type
        content_lower = content.lower()
        if any(word in content_lower for word in ['beam', 'column', 'steel']):
            schedule_info['type'] = 'structural_schedule'
        elif any(word in content_lower for word in ['door', 'window', 'opening']):
            schedule_info['type'] = 'opening_schedule'
        elif any(word in content_lower for word in ['room', 'space', 'area']):
            schedule_info['type'] = 'room_schedule'
        
        # Extract structured data (basic parsing)
        for line in lines[:10]:  # Process first 10 lines
            if line.strip():
                # Look for key-value patterns
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        schedule_info['key_values'][parts[0].strip()] = parts[1].strip()
        
        return schedule_info
    
    def _parse_title_block(self, content: str) -> Dict[str, str]:
        """Parse title block content for sheet metadata"""
        if not content:
            return {}
        
        title_info = {
            'drawing_title': '',
            'drawing_number': '',
            'scale': '',
            'date': '',
            'revision': ''
        }
        
        # Extract common title block information
        lines = content.split('\n') if content else []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Scale information
            scale_match = re.search(r'(\d+/?\d*[\"\u2033]?\s*=\s*\d+[\'\u2032]-\d+[\"\u2033]?)', line, re.IGNORECASE)
            if scale_match:
                title_info['scale'] = scale_match.group(1)
            
            # Drawing numbers (like A3.1, S7.5, etc.)
            dwg_num_match = re.search(r'\b[A-Z]\d+(?:\.\d+)?\b', line)
            if dwg_num_match:
                title_info['drawing_number'] = dwg_num_match.group(0)
            
            # Dates
            date_match = re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', line)
            if date_match:
                title_info['date'] = date_match.group(0)
        
        return title_info
    
    def _create_intelligent_links(self, elements: Dict[str, List]) -> Dict[str, List]:
        """Create intelligent cross-links between all element types"""
        links = {
            'dimension_to_drawing': [],
            'text_to_drawing': [],
            'schedule_to_elements': [],
            'title_to_content': [],
            'semantic_clusters': []
        }
        
        # Get element lists
        dimensions = elements.get('dimension_line', [])
        drawings = elements.get('drawing_area', [])
        texts = elements.get('text_block', [])
        schedules = elements.get('schedule_table', [])
        titles = elements.get('title_block', [])
        
        # Link dimensions to drawings (spatial)
        for dim in dimensions:
            dim_spatial = dim.get('spatial', {})
            best_match = None
            best_proximity = 0
            
            for drawing in drawings:
                drawing_spatial = drawing.get('spatial', {})
                proximity = self.rag_system._calculate_proximity(dim_spatial, drawing_spatial)
                
                if proximity > best_proximity:
                    best_proximity = proximity
                    best_match = drawing
            
            if best_match and best_proximity > 0.2:
                links['dimension_to_drawing'].append({
                    'dimension': dim,
                    'linked_drawing': best_match,
                    'proximity_score': best_proximity,
                    'measurements': self.rag_system._extract_measurements(dim.get('content', ''))
                })
        
        # Link text blocks to drawings (spatial + semantic)
        for text in texts:
            text_spatial = text.get('spatial', {})
            text_content = text.get('content', '')
            
            # Find spatially close drawings
            for drawing in drawings:
                drawing_spatial = drawing.get('spatial', {})
                proximity = self.rag_system._calculate_proximity(text_spatial, drawing_spatial)
                
                if proximity > 0.3:
                    # Also check semantic relevance
                    semantic_score = self._calculate_semantic_relevance(text_content, drawing.get('content', ''))
                    
                    links['text_to_drawing'].append({
                        'text_block': text,
                        'linked_drawing': drawing,
                        'proximity_score': proximity,
                        'semantic_score': semantic_score,
                        'combined_score': proximity * 0.7 + semantic_score * 0.3
                    })
        
        # Link schedules to all related elements
        for schedule in schedules:
            schedule_content = schedule.get('content', '')
            schedule_data = self._parse_schedule_content(schedule_content)
            
            # Find elements referenced in schedule
            referenced_elements = []
            
            # Check if dimensions, text, or drawings reference this schedule
            for element_type, element_list in elements.items():
                for element in element_list:
                    element_content = element.get('content', '')
                    if self._has_schedule_reference(element_content, schedule_content):
                        referenced_elements.append({
                            'element_type': element_type,
                            'element': element,
                            'reference_strength': 0.8
                        })
            
            if referenced_elements:
                links['schedule_to_elements'].append({
                    'schedule': schedule,
                    'schedule_data': schedule_data,
                    'referenced_elements': referenced_elements
                })
        
        # Link titles to content (semantic)
        for title in titles:
            title_content = title.get('content', '')
            title_info = self._parse_title_block(title_content)
            
            # Find all elements that semantically relate to this title
            related_content = []
            
            for element_type, element_list in elements.items():
                if element_type == 'title_block':
                    continue
                    
                for element in element_list:
                    semantic_score = self._calculate_semantic_relevance(
                        title_content, 
                        element.get('content', '')
                    )
                    
                    if semantic_score > 0.3:
                        related_content.append({
                            'element_type': element_type,
                            'element': element,
                            'semantic_score': semantic_score
                        })
            
            links['title_to_content'].append({
                'title': title,
                'title_info': title_info,
                'related_content': sorted(related_content, key=lambda x: x['semantic_score'], reverse=True)[:10]
            })
        
        return links
    
    def _calculate_semantic_relevance(self, text1: str, text2: str) -> float:
        """Calculate semantic relevance between two text pieces"""
        if not text1 or not text2:
            return 0.0
        
        # Simple but effective semantic scoring
        text1_words = set(re.findall(r'\b\w+\b', text1.lower()))
        text2_words = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not text1_words or not text2_words:
            return 0.0
        
        # Jaccard similarity with architectural keyword weighting
        intersection = text1_words & text2_words
        union = text1_words | text2_words
        
        base_score = len(intersection) / len(union)
        
        # Boost score for architectural keywords
        arch_keywords = {'beam', 'column', 'wall', 'floor', 'ceiling', 'door', 'window', 
                        'steel', 'concrete', 'dimension', 'specification', 'detail'}
        
        arch_matches = len(intersection & arch_keywords)
        boost = min(0.3, arch_matches * 0.1)
        
        return min(1.0, base_score + boost)
    
    def _has_schedule_reference(self, element_content: str, schedule_content: str) -> bool:
        """Check if element references a schedule"""
        if not element_content or not schedule_content:
            return False
        
        # Look for common schedule references
        schedule_indicators = ['schedule', 'table', 'spec', 'grade', 'type']
        
        element_lower = element_content.lower()
        schedule_lower = schedule_content.lower()
        
        # Check for shared key terms
        shared_terms = 0
        for indicator in schedule_indicators:
            if indicator in element_lower and indicator in schedule_lower:
                shared_terms += 1
        
        return shared_terms >= 2
    
    def _extract_domain_knowledge(self, elements: Dict[str, List]) -> Dict[str, set]:
        """Extract domain-specific knowledge from all elements"""
        knowledge = {
            'measurements': set(),
            'materials': set(), 
            'specifications': set(),
            'room_names': set(),
            'equipment': set()
        }
        
        for element_type, element_list in elements.items():
            for element in element_list:
                content = element.get('content', '')
                
                if content:
                    # Extract measurements
                    measurements = self.rag_system._extract_measurements(content)
                    knowledge['measurements'].update(measurements)
                    
                    # Extract other structured info
                    structured = self._extract_structured_info(content)
                    for key, values in structured.items():
                        if key in knowledge:
                            knowledge[key].update(values)
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in knowledge.items()}
    
    def intelligent_query_parse(self, query: str) -> Dict[str, Any]:
        """
        Use Qwen to intelligently parse architectural queries
        """
        
        # Prompt engineering for architectural query understanding
        parse_prompt = f"""You are an expert architectural drawing analyst. Parse this query to understand the user's intent.

Query: "{query}"

Analyze and respond in this exact JSON format:
{{
  "intent_type": "dimensions|materials|specifications|structural|electrical|mechanical|general",
  "confidence": "high|medium|low", 
  "key_elements": ["list", "of", "key", "terms"],
  "search_focus": "text_heavy|image_heavy|balanced",
  "expected_answer_type": "measurements|specifications|locations|descriptions|lists",
  "architectural_domain": "structural|architectural|mechanical|electrical|civil",
  "complexity": "simple|moderate|complex"
}}

Be precise and architectural-focused. Consider building codes, construction terminology, and technical specifications."""

        if self.query_parser_llm == "openai":
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an architectural drawing analysis expert. Respond ONLY with valid JSON."},
                        {"role": "user", "content": parse_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                    
            except Exception as e:
                print(f"OpenAI parsing error: {e}")
        
        # Fallback to rule-based parsing
        return self._rule_based_query_parse(query)
    
    def _rule_based_query_parse(self, query: str) -> Dict[str, Any]:
        """Fallback rule-based query parsing"""
        query_lower = query.lower()
        
        # Determine intent
        if any(word in query_lower for word in ['dimension', 'size', 'width', 'height', 'length']):
            intent = "dimensions"
            search_focus = "text_heavy"
        elif any(word in query_lower for word in ['material', 'steel', 'concrete', 'grade']):
            intent = "materials"
            search_focus = "balanced"
        elif any(word in query_lower for word in ['beam', 'column', 'structural']):
            intent = "structural"
            search_focus = "balanced"
        else:
            intent = "general"
            search_focus = "balanced"
        
        return {
            "intent_type": intent,
            "confidence": "medium",
            "key_elements": re.findall(r'\b\w+\b', query_lower),
            "search_focus": search_focus,
            "expected_answer_type": "descriptions",
            "architectural_domain": "architectural",
            "complexity": "moderate"
        }
    
    def intelligent_response_generate(self, query: str, context: IntelligentContext, 
                                    parsed_query: Dict[str, Any]) -> str:
        """
        Use Qwen to generate intelligent architectural responses
        """
        
        # Prepare rich context for response generation
        context_summary = self._prepare_context_summary(context)
        
        # Architectural expertise prompt with rich context
        response_prompt = f"""You are a senior architectural consultant with expertise in construction documents, building codes, and technical specifications. Answer this technical query with precision and authority.

QUERY: {query}
QUERY ANALYSIS: {json.dumps(parsed_query, indent=2)}

RELEVANT TECHNICAL CONTEXT:
{context_summary}

INSTRUCTIONS:
1. Provide a precise, technical answer based ONLY on the provided context
2. Include specific measurements, materials, specifications, and sheet references
3. If dimensions are involved, include units and verify accuracy
4. Reference specific drawing sheets and elements when available
5. Acknowledge any limitations or ambiguities in the available information
6. Use professional architectural terminology
7. Be confident but honest about what the drawings show vs. don't show

TECHNICAL RESPONSE:"""

        if self.response_generator_llm == "openai":
            try:
                response = openai.chat.completions.create(
                    model="gpt-4",  # Use GPT-4 for better architectural analysis
                    messages=[
                        {"role": "system", "content": "You are a senior architectural consultant with expertise in construction documents, building codes, and technical specifications. Provide precise, technical answers."},
                        {"role": "user", "content": response_prompt}
                    ],
                    max_tokens=600,
                    temperature=0.3
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"OpenAI response error: {e}")
                # Fallback to GPT-3.5 if GPT-4 fails
                try:
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an architectural consultant. Provide technical answers based on construction documents."},
                            {"role": "user", "content": response_prompt}
                        ],
                        max_tokens=500,
                        temperature=0.3
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e2:
                    print(f"OpenAI fallback error: {e2}")
        
        # Fallback to enhanced rule-based response
        return self._generate_intelligent_fallback(query, context, parsed_query)
    
    def _prepare_context_summary(self, context: IntelligentContext) -> str:
        """Prepare rich context summary for LLM"""
        summary_parts = []
        
        # Primary content
        if context.primary_content:
            summary_parts.append(f"PRIMARY: {context.primary_content[:200]}")
        
        # Related drawings with spatial info
        if context.related_drawings:
            summary_parts.append("DRAWINGS:")
            for i, drawing in enumerate(context.related_drawings[:3]):
                summary_parts.append(f"  {i+1}. {drawing.get('label', 'Unknown')} (confidence: {drawing.get('ocr_confidence', 0):.2f})")
                if drawing.get('content'):
                    summary_parts.append(f"     Content: {drawing['content'][:100]}")
        
        # Related dimensions with measurements
        if context.related_dimensions:
            summary_parts.append("DIMENSIONS:")
            for i, dim in enumerate(context.related_dimensions[:3]):
                measurements = dim.get('measurements', [])
                summary_parts.append(f"  {i+1}. {dim.get('dimension_text', '')} | Measurements: {measurements}")
        
        # Related text blocks
        if context.related_text_blocks:
            summary_parts.append("TEXT BLOCKS:")
            for i, text in enumerate(context.related_text_blocks[:3]):
                summary_parts.append(f"  {i+1}. {text.get('content', '')[:100]}")
        
        # Schedules and specifications
        if context.related_schedules:
            summary_parts.append("SCHEDULES/SPECS:")
            for i, schedule in enumerate(context.related_schedules[:2]):
                summary_parts.append(f"  {i+1}. {schedule.get('content', '')[:100]}")
        
        return "\n".join(summary_parts)
    
    def _generate_intelligent_fallback(self, query: str, context: IntelligentContext, 
                                     parsed_query: Dict[str, Any]) -> str:
        """Enhanced rule-based response when LLM not available"""
        
        response_parts = []
        
        # Start with intent-specific response
        intent = parsed_query.get('intent_type', 'general')
        
        if intent == 'dimensions' and context.related_dimensions:
            response_parts.append("📏 DIMENSIONAL INFORMATION:")
            for dim in context.related_dimensions[:3]:
                measurements = dim.get('measurements', [])
                if measurements:
                    response_parts.append(f"  • {', '.join(measurements)} - {dim.get('dimension_text', '')[:60]}")
        
        elif intent == 'materials' and (context.related_schedules or context.related_text_blocks):
            response_parts.append("🔩 MATERIAL SPECIFICATIONS:")
            
            # Check schedules first
            for schedule in context.related_schedules[:2]:
                content = schedule.get('content', '')
                materials = re.findall(r'\b(?:Grade\s*\d+|A\d{3}|steel|concrete|f[\'\u2032]c\s*=?\s*\d+)\b', content, re.IGNORECASE)
                if materials:
                    response_parts.append(f"  • {', '.join(set(materials))} (from schedule)")
            
            # Check text blocks
            for text in context.related_text_blocks[:2]:
                content = text.get('content', '')
                materials = re.findall(r'\b(?:Grade\s*\d+|A\d{3}|steel|concrete)\b', content, re.IGNORECASE)
                if materials:
                    response_parts.append(f"  • {', '.join(set(materials))} (from notes)")
        
        else:
            # General response with all available context
            response_parts.append("📋 DRAWING ANALYSIS:")
            
            if context.related_drawings:
                response_parts.append(f"  • Found in {len(context.related_drawings)} drawing areas")
            
            if context.related_dimensions:
                all_measurements = []
                for dim in context.related_dimensions:
                    all_measurements.extend(dim.get('measurements', []))
                if all_measurements:
                    response_parts.append(f"  • Measurements: {', '.join(list(set(all_measurements))[:5])}")
            
            if context.related_text_blocks:
                response_parts.append(f"  • Related text blocks: {len(context.related_text_blocks)}")
        
        # Add source information
        if context.related_drawings:
            sheet_ids = set()
            for drawing in context.related_drawings:
                sheet_id = drawing.get('metadata', {}).get('sheet_id')
                if sheet_id:
                    sheet_ids.add(sheet_id)
            
            if sheet_ids:
                response_parts.append(f"\n📄 Source sheets: {', '.join(list(sheet_ids)[:3])}")
        
        # Add confidence
        confidence_text = f"Confidence: {context.confidence_score:.2f}"
        response_parts.append(f"\n🎯 {confidence_text}")
        
        return "\n".join(response_parts)
    
    def intelligent_search(self, query: str) -> IntelligentContext:
        """
        Perform intelligent search that links ALL related information
        """
        # Step 1: Parse query intelligently
        parsed_query = self.intelligent_query_parse(query)
        print(f"🧠 Parsed query: {parsed_query['intent_type']} ({parsed_query['confidence']})")
        
        # Step 2: Retrieve based on intelligent analysis
        if parsed_query['search_focus'] == 'text_heavy':
            text_results = self.rag_system.retrieve_text(query, k=15)
            image_results = self.rag_system.retrieve_images_by_caption(query, k=8)
            fusion_weights = (0.8, 0.2)
        elif parsed_query['search_focus'] == 'image_heavy':
            text_results = self.rag_system.retrieve_text(query, k=8)
            image_results = self.rag_system.retrieve_images_by_caption(query, k=15)
            fusion_weights = (0.3, 0.7)
        else:
            text_results = self.rag_system.retrieve_text(query, k=12)
            image_results = self.rag_system.retrieve_images_by_caption(query, k=10)
            fusion_weights = (0.65, 0.35)
        
        # Step 3: Fuse results
        fused_results = self.rag_system.fuse_results(
            text_results, image_results,
            w_text=fusion_weights[0], w_img=fusion_weights[1], top=12
        )
        
        # Step 4: Build intelligent context by linking ALL related information
        return self._build_intelligent_context(query, fused_results, parsed_query)
    
    def _build_intelligent_context(self, query: str, fused_results: List[Dict], 
                                 parsed_query: Dict[str, Any]) -> IntelligentContext:
        """Build rich context with ALL related information linked"""
        
        if not fused_results:
            return IntelligentContext(
                primary_content="",
                related_drawings=[],
                related_text_blocks=[],
                related_dimensions=[],
                related_schedules=[],
                title_context=[],
                spatial_relationships={},
                semantic_links=[],
                confidence_score=0.0
            )
        
        # Primary content from top result
        primary_result = fused_results[0]
        primary_content = primary_result['doc'] or f"Visual element from {primary_result['meta'].get('sheet_id', 'Unknown')}"
        
        # Categorize and link related results
        related_drawings = []
        related_text_blocks = []
        related_dimensions = []
        related_schedules = []
        title_context = []
        
        for result in fused_results:
            meta = result['meta']
            label = meta.get('label', 'Unknown')
            content = result['doc']
            
            element_info = {
                'content': content,
                'metadata': meta,
                'confidence_score': result['score'],
                'ocr_confidence': meta.get('ocr_confidence', 0)
            }
            
            # Categorize by label type
            if label == 'drawing_area':
                related_drawings.append(element_info)
            elif label == 'text_block':
                related_text_blocks.append(element_info)
            elif label == 'dimension_line':
                measurements = self.rag_system._extract_measurements(content)
                element_info['measurements'] = measurements
                element_info['dimension_text'] = content
                related_dimensions.append(element_info)
            elif label == 'schedule_table':
                related_schedules.append(element_info)
            elif label == 'title_block':
                title_context.append(element_info)
        
        # Calculate overall confidence
        confidence_scores = [r['score'] for r in fused_results[:5]]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # For dimension/measurement queries, enhance with spatial context
        if parsed_query['intent_type'] == 'dimensions' and related_dimensions:
            # Get additional spatial context for measurements
            for dim in related_dimensions:
                sheet_id = dim['metadata'].get('sheet_id')
                spatial_data = self.rag_system._parse_spatial_data(dim['metadata'].get('spatial', '{}'))
                
                if sheet_id and spatial_data:
                    related_content = self.rag_system._get_spatially_related_content(sheet_id, spatial_data)
                    dim['spatial_context'] = related_content
        
        return IntelligentContext(
            primary_content=primary_content,
            related_drawings=related_drawings,
            related_text_blocks=related_text_blocks,
            related_dimensions=related_dimensions,
            related_schedules=related_schedules,
            title_context=title_context,
            spatial_relationships={},
            semantic_links=[],
            confidence_score=avg_confidence
        )
    
    def handle_document_questions(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Handle document-level questions directly without vector search
        
        Questions like: "How many drawings?", "What types?", "Show me all sheets"
        """
        query_lower = query.lower()
        
        # Check if this is a document-level question
        document_indicators = [
            'how many drawing', 'how many sheet', 'total drawing', 'total sheet',
            'what types', 'list all', 'show me all', 'document set', 'drawings available',
            'inventory', 'catalog', 'overview'
        ]
        
        if not any(indicator in query_lower for indicator in document_indicators):
            return None  # Not a document question
        
        print("📊 Handling document-level question directly")
        
        # Get comprehensive inventory
        inventory = self.rag_system.get_drawing_inventory()
        
        # Generate direct answer based on question type
        if any(word in query_lower for word in ['how many', 'total', 'count']):
            if 'drawing' in query_lower or 'sheet' in query_lower:
                return {
                    'query': query,
                    'answer': f"📊 **Document Inventory**: This architectural document set contains **{inventory['total_sheets']} drawing sheets** with **{inventory['total_elements']} total elements**.\\n\\n**Drawing Types:**\\n" + 
                             "\\n".join([f"• {dtype}: {count} elements" for dtype, count in inventory['drawing_types'].items()]) +
                             f"\\n\\n**Key Statistics:**\\n• {inventory['summary']['sheets_with_dimensions']} sheets have dimensional information\\n• {inventory['summary']['sheets_with_schedules']} sheets contain schedules/tables",
                    'confidence': 1.0,  # High confidence for inventory questions
                    'parsed_query': {'intent_type': 'inventory', 'confidence': 'high'},
                    'context_summary': {
                        'total_sheets': inventory['total_sheets'],
                        'total_elements': inventory['total_elements'],
                        'drawing_types': len(inventory['drawing_types'])
                    },
                    'source_sheets': list(inventory['sheets_detail'].keys())[:10]
                }
        
        return None  # Couldn't handle as document question
    
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Main intelligent query interface with document question handling
        """
        print(f"🤔 Processing: {query}")
        
        # First check if this is a document-level question
        doc_response = self.handle_document_questions(query)
        if doc_response:
            return doc_response
        
        # Step 1: Intelligent search with full context linking
        context = self.intelligent_search(query)
        
        # Step 2: Parse query for response optimization  
        parsed_query = self.intelligent_query_parse(query)
        
        # Step 3: Generate intelligent response
        response = self.intelligent_response_generate(query, context, parsed_query)
        
        return {
            'query': query,
            'answer': response,
            'confidence': context.confidence_score,
            'parsed_query': parsed_query,
            'context_summary': {
                'drawings_found': len(context.related_drawings),
                'dimensions_found': len(context.related_dimensions),
                'text_blocks_found': len(context.related_text_blocks),
                'schedules_found': len(context.related_schedules),
                'titles_found': len(context.title_context)
            },
            'measurements': [dim.get('measurements', []) for dim in context.related_dimensions],
            'source_sheets': list(set([
                d['metadata'].get('sheet_id') for d in 
                context.related_drawings + context.related_text_blocks + context.related_dimensions
                if d['metadata'].get('sheet_id')
            ]))
        }

# Test the intelligent system
if __name__ == "__main__":
    print("🧠 Testing Intelligent Architectural System")
    print("=" * 60)
    
    system = IntelligentArchSystem()
    
    # Test intelligent queries
    test_queries = [
        "How many drawings are in this document set?",
        "What structural steel specifications are shown?", 
        "What are the room dimensions for mechanical spaces?",
        "Show me beam connection details with measurements"
    ]
    
    for query in test_queries[:2]:  # Test first 2
        print(f"\n{'='*50}")
        result = system.ask(query)
        
        print(f"❓ {result['query']}")
        print(f"🤖 {result['answer']}")
        print(f"📊 Context: {result['context_summary']}")
        print(f"📏 Measurements: {[m for measurements in result['measurements'] for m in measurements][:5]}")
        print(f"📄 Source sheets: {result['source_sheets'][:3]}")
    
    print(f"\n✅ Intelligent system testing complete!")