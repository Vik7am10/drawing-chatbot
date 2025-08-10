import chromadb
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import open_clip
import torch
from PIL import Image
import requests
from io import BytesIO
import base64
import os
import sqlite3
import re
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

class ArchRAGSystem:
    def __init__(self, vectordb_path="./vectordb_extracted/vectordb_bge_openclip-v2"):
        """
        Initialize the architectural RAG system - EXACT logic from evaluation pipeline
        
        Key specs from evaluation pipeline:
        - Text: BGE "BAAI/bge-base-en-v1.5" with 768 dimensions (normalized)
        - Image: OpenCLIP "ViT-B-32" with "laion2b_s34b_b79k" pretrained, 512 dimensions
        - Collections: "arch_text" and "arch_visual"
        """
        self.vectordb_path = vectordb_path
        
        # Disable telemetry like in evaluation pipeline
        os.environ["CHROMA_TELEMETRY_DISABLED"] = "TRUE"
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=vectordb_path)
        
        # Device setup like evaluation pipeline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", self.device)
        
        # Get collections with EXACT names from evaluation pipeline
        self.text_collection = None
        self.image_collection = None
        
        try:
            self.text_collection = self.chroma_client.get_collection("arch_text")
            print(f"✓ Text collection: {self.text_collection.count()} documents")
        except Exception as e:
            print(f"⚠ Text collection not found: {e}")
            
        try:
            self.image_collection = self.chroma_client.get_collection("arch_visual") 
            print(f"✓ Image collection: {self.image_collection.count()} documents")
        except Exception as e:
            print(f"⚠ Image collection not found: {e}")
        
        # Detect image collection dimension like evaluation pipeline
        self.img_dim = self._detect_image_dim()
        print(f"Image collection dimension: {self.img_dim}")
        
        # Models for encoding - PRELOAD during initialization
        print("🔄 Preloading BGE text encoder...")
        self.text_model = None
        self.clip_model = None
        self.tokenizer = None
        self.clip_preprocess = None
        
        # Preload text encoder
        self.load_text_encoder()
        
        # Preload image encoder if we have image collection
        if self.image_collection is not None and self.img_dim is not None:
            print("🔄 Preloading OpenCLIP image encoder...")
            self.load_image_encoder()

    
    def _fetch_all(self, collection, include=('metadatas','documents'), batch=500, where=None):
        out = {k: [] for k in include}
        out['ids'] = []
        offset = 0
        while True:
            chunk = collection.get(include=list(include), where=where, limit=batch, offset=offset)
            ids = chunk.get('ids', [])
            if not ids:
                break
            out['ids'].extend(ids)
            for k in include:
                out[k].extend(chunk.get(k, []))
            offset += len(ids)
        return out

    def _norm(self, s): 
        return (s or "").strip().upper()

    def _detect_image_dim(self):
        """Detect image collection dimension - EXACT same logic as evaluation pipeline"""
        if self.image_collection is None:
            return None
            
        def accepts_dim(col, d):
            try:
                col.query(query_embeddings=[[0.0]*d], n_results=1)
                return True
            except Exception:
                return False
        
        if accepts_dim(self.image_collection, 512):
            return 512
        elif accepts_dim(self.image_collection, 768): 
            return 768
        else:
            return None
        
    def load_text_encoder(self, model_name="BAAI/bge-base-en-v1.5"):
        """Load BGE text encoder - EXACT same as evaluation pipeline"""
        if self.text_model is None:
            print(f"Loading text encoder: {model_name}")
            self.text_model = SentenceTransformer(model_name, device=self.device)
            assert self.text_model.get_sentence_embedding_dimension() == 768
        return self.text_model
        
    def load_image_encoder(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        """Load OpenCLIP image encoder - EXACT same as evaluation pipeline"""
        if self.clip_model is None:
            print(f"Loading image encoder: {model_name} ({pretrained})")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            # Verify dimension like evaluation pipeline
            with torch.no_grad():
                dummy = torch.randn(1,3,224,224, device=self.device)
                assert self.clip_model.encode_image(dummy).shape[-1] == 512
        return self.clip_model, self.clip_preprocess
    
    def normalize_units(self, q: str) -> str:
        """Unit normalization - EXACT same logic as evaluation pipeline"""
        # Match feet/inches formats like 6'4" or 6′4″
        FTIN = re.compile(r"(\d+)[\'′]\s*(\d+)?(?:[\"\u2033])?")
        IN_ONLY = re.compile(r"(\d+)\s*(?:in|inch|inches)\b", re.I)
        MM = re.compile(r"(\d+(?:\.\d+)?)\s*mm\b", re.I)
        
        def ftin_to_mm(m):
            ft = int(m.group(1))
            inch = int(m.group(2) or 0)
            total_in = ft * 12 + inch
            total_mm = round(total_in * 25.4)
            return f"{ft}'{inch}\" [{total_mm} mm]"
        
        q = FTIN.sub(ftin_to_mm, q)
        q = IN_ONLY.sub(lambda m: f"{m.group(1)} in [{round(int(m.group(1))*25.4)} mm]", q)
        q = MM.sub(lambda m: f"{m.group(1)} mm", q)
        return q
    
    def encode_text_query(self, query_text):
        """
        Encode text query using BGE model - EXACT same logic as evaluation pipeline
        
        CRITICAL: BGE embeddings are normalized and have dimension 768
        """
        if self.text_model is None:
            self.load_text_encoder()
            
        # Normalize units first like evaluation pipeline
        normalized_query = self.normalize_units(query_text)
        
        # BGE produces normalized embeddings - exact same call as evaluation pipeline
        embedding = self.text_model.encode([normalized_query], normalize_embeddings=True, convert_to_numpy=True)
        
        # Verify dimensions (from evaluation pipeline)
        if embedding.shape[1] != 768:
            raise ValueError(f"Expected BGE embedding dimension 768, got {embedding.shape[1]}")
            
        return embedding
    
    def encode_caption_512(self, caption: str):
        """Encode caption using OpenCLIP text encoder - EXACT same as evaluation pipeline"""
        if self.clip_model is None:
            self.load_image_encoder()
            
        # Normalize units first like evaluation pipeline
        normalized_caption = self.normalize_units(caption)
        
        toks = self.tokenizer([normalized_caption]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device=='cuda')):
            tfeat = self.clip_model.encode_text(toks)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        v = tfeat.detach().cpu().numpy()
        assert v.shape[1] == 512
        return v
    
    def encode_image_path_512(self, path: str):
        """Encode image from path - EXACT same as evaluation pipeline"""
        if self.clip_model is None:
            self.load_image_encoder()
            
        img = Image.open(path).convert("RGB")
        x = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device=='cuda')):
            v = self.clip_model.encode_image(x)
            v = v / v.norm(dim=-1, keepdim=True)
        v = v.detach().cpu().numpy()
        assert v.shape[1] == 512
        return v
    
    def decode_base64_image(self, base64_string, output_path=None):
        """
        Decode base64 encoded image data - EXACT same logic as evaluation pipeline
        
        IMPORTANT: Handle decoding carefully - images may be stored as base64 in metadata
        """
        try:
            # Remove data URL prefix if present (from evaluation pipeline)
            if base64_string.startswith('data:image/'):
                base64_string = base64_string.split(',')[1]
                
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            if output_path:
                # Save image
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                print(f"Image saved to {output_path}")
                
            # Return as PIL Image for processing
            return Image.open(BytesIO(image_data))
            
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None
    
    def retrieve_text(self, query: str, k: int = 10, where=None):
        """Search text collection - EXACT same as evaluation pipeline"""
        if self.text_collection is None:
            return {"error": "Text collection not available"}
            
        v = self.encode_text_query(query)
        return self.text_collection.query(
            query_embeddings=v.tolist(), 
            n_results=k,
            where=(where or {"label": {"$ne": "drawing_area"}}),
            include=['metadatas', 'documents', 'distances']
        )
    
    def retrieve_images_by_caption(self, query: str, k: int = 8):
        """Search images by caption - EXACT same as evaluation pipeline"""
        if self.image_collection is None:
            return {"error": "Image collection not available"}
            
        if self.img_dim == 512:
            qv = self.encode_caption_512(query)
        else:  # img_dim == 768: the \"image\" collection was built with a 768-d text encoder
            qv = self.encode_text_query(query)
        return self.image_collection.query(
            query_embeddings=qv.tolist(), 
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
    
    def retrieve_images_by_image(self, path: str, k: int = 8):
        """Search images by image - EXACT same as evaluation pipeline"""
        if self.image_collection is None:
            return {"error": "Image collection not available"}
            
        qv = self.encode_image_path_512(path)
        return self.image_collection.query(
            query_embeddings=qv.tolist(), 
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
    
    def fuse_results(self, text_res: Dict[str,Any], img_res: Dict[str,Any], w_text=0.65, w_img=0.35, top=10):
        """Fuse results - EXACT same logic as evaluation pipeline"""
        # Expect .get("ids"/"documents"/"metadatas")[0] lists
        items = []
        for source, res, w in [("text", text_res, w_text), ("image", img_res, w_img)]:
            # ids = res.get("ids",[[]])[0]  # Skip ids since not available
            docs = res.get("documents",[[]])[0]
            metas = res.get("metadatas",[[]])[0]
            dists = res.get("distances",[[]])[0]  # cosine distance; lower is better
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                id_ = f"{source}_{i}"  # generate simple id
                score = (1.0 - dist) * w  # convert to similarity-ish
                items.append((id_, doc, meta or {}, score, source))
        
        # Merge by id, keep max score per id
        agg = {}
        for id_, doc, meta, score, src in items:
            if id_ not in agg or score > agg[id_]["score"]:
                agg[id_] = {"id": id_, "doc": doc, "meta": meta, "score": score, "src": src}
        ranked = sorted(agg.values(), key=lambda x: x["score"], reverse=True)[:top]
        return ranked
    
    def show_text_res(self, res, top=5):
        """Display text results - EXACT same as evaluation pipeline"""
        docs = res.get("documents",[[]])[0]
        metas = res.get("metadatas",[[]])[0]
        for i,(d,m) in enumerate(zip(docs, metas), start=1):
            print(f"{i}. {m.get('sheet_id')} | {m.get('label')} | OCR {m.get('ocr_confidence',0):.2f}")
            print("   ", (d or "").replace("\n"," ")[:180] + ("..." if d and len(d)>180 else ""))
        print("-"*80)
    
    def show_image_res(self, res, top=5):
        """Display image results - EXACT same as evaluation pipeline"""
        docs = res.get("documents",[[]])[0]
        metas = res.get("metadatas",[[]])[0]
        for i,(p,m) in enumerate(zip(docs, metas), start=1):
            print(f"{i}. {m.get('sheet_id')} | {m.get('label')} | path={p}")
        print("-"*80)
    
    def show_fused(self, items, top=10):
        """Display fused results - EXACT same as evaluation pipeline"""
        for i,it in enumerate(items[:top], start=1):
            m = it["meta"]
            print(f"{i}. src={it['src']} | score={it['score']:.3f} | {m.get('sheet_id')} | {m.get('label')} | id={it['id']}")
            print("   ", (it["doc"] or "").replace("\n"," ")[:180] + ("..." if it["doc"] and len(it["doc"])>180 else ""))
    
    def inspect_database(self):
        """Inspect the database structure"""
        info = {}
        
        if self.text_collection:
            sample = self.text_collection.get(limit=3, include=['metadatas', 'documents'])
            info['text_collection'] = {
                'count': self.text_collection.count(),
                'sample_documents': sample['documents'][:2] if sample['documents'] else [],
                'sample_metadata': sample['metadatas'][:2] if sample['metadatas'] else []
            }
            
        if self.image_collection:
            sample = self.image_collection.get(limit=3, include=['metadatas', 'documents'])
            info['image_collection'] = {
                'count': self.image_collection.count(),
                'sample_documents': sample['documents'][:2] if sample['documents'] else [],
                'sample_metadata': sample['metadatas'][:2] if sample['metadatas'] else []
            }
            
        return info

    def get_drawing_inventory(self):
        if self.text_collection is None:
            return {"error": "Text collection not available"}

        # Pull EVERYTHING once (no 1000 cap, no fake pagination via query)
        all_docs = []
        pulled = self._fetch_all(self.text_collection, include=('metadatas','documents'))
        docs = pulled.get('documents', [])
        metas = pulled.get('metadatas', [])
        for doc, meta in zip(docs, metas):
            all_docs.append({'doc': doc, 'meta': meta or {}})

        sheets = {}
        drawing_types = {}
        labels_count = {}

        for item in all_docs:
            meta = item['meta']
            sheet_id = meta.get('sheet_id', 'Unknown')
            label = meta.get('label', 'Unknown')

            if sheet_id not in sheets:
                sheets[sheet_id] = {
                    'labels': {},
                    'total_elements': 0,
                    'has_dimensions': False,
                    'has_text_blocks': False,
                    'has_schedules': False,
                    'drawing_type': self._infer_drawing_type(sheet_id)
                }

            sheets[sheet_id]['labels'][label] = sheets[sheet_id]['labels'].get(label, 0) + 1
            sheets[sheet_id]['total_elements'] += 1
            if label == 'dimension_line':
                sheets[sheet_id]['has_dimensions'] = True
            elif label in ['text_block', 'title_block']:
                sheets[sheet_id]['has_text_blocks'] = True
            elif label == 'schedule_table':
                sheets[sheet_id]['has_schedules'] = True

            drawing_type = sheets[sheet_id]['drawing_type']
            drawing_types[drawing_type] = drawing_types.get(drawing_type, 0) + 1
            labels_count[label] = labels_count.get(label, 0) + 1

        return {
            'total_sheets': len(sheets),
            'total_elements': len(all_docs),
            'sheets_detail': sheets,  # ← no [:20] truncation
            'drawing_types': drawing_types,
            'label_distribution': labels_count,
            'summary': {
                'sheets_with_dimensions': sum(1 for s in sheets.values() if s['has_dimensions']),
                'sheets_with_schedules': sum(1 for s in sheets.values() if s['has_schedules']),
                'most_common_labels': sorted(labels_count.items(), key=lambda x: x[1], reverse=True)[:10]
            }
        }

    
    def _infer_drawing_type(self, sheet_id: str) -> str:
        """Infer drawing type from sheet ID"""
        sheet_lower = sheet_id.lower()
        
        if any(word in sheet_lower for word in ['floor', 'plan']):
            return 'Floor Plan'
        elif any(word in sheet_lower for word in ['elevation', 'section']):
            return 'Elevation/Section'
        elif any(word in sheet_lower for word in ['detail', 'connection']):
            return 'Detail Drawing'
        elif any(word in sheet_lower for word in ['steel', 'structural']):
            return 'Structural Drawing'
        elif any(word in sheet_lower for word in ['mechanical', 'hvac', 'plumbing']):
            return 'MEP Drawing'
        elif any(word in sheet_lower for word in ['electrical']):
            return 'Electrical Drawing'
        elif any(word in sheet_lower for word in ['site', 'civil']):
            return 'Site/Civil Drawing'
        else:
            return 'General Drawing'
    
    def get_spatial_links(self, target_element_id: str = None, sheet_id: str = None):
        if self.text_collection is None:
            return {"error": "Text collection not available"}

        where = {"sheet_id": sheet_id} if sheet_id else None
        pulled = self._fetch_all(self.text_collection, include=('metadatas','documents'), where=where)
        docs = pulled.get('documents', [])
        metas = pulled.get('metadatas', [])

        sheet_elements = {}
        for doc, meta in zip(docs, metas):
            meta = meta or {}
            sheet = meta.get('sheet_id', 'Unknown')
            label = meta.get('label', 'Unknown')
            if label not in ['dimension_line', 'drawing_area', 'text_block']:
                continue
            sheet_elements.setdefault(sheet, {'dimension_lines': [], 'drawing_areas': [], 'text_blocks': []})
            element_info = {
                'doc': doc,
                'meta': meta,
                'spatial': self._parse_spatial_data(meta.get('spatial', '{}')),
                'label': label
            }
            sheet_elements[sheet][label + 's' if label != 'text_block' else 'text_blocks'].append(element_info)

        spatial_links = {}
        for sheet, elements in sheet_elements.items():
            spatial_links[sheet] = {
                'linked_dimensions': [],
                'drawing_info': {},
                'measurements_found': []
            }
            for dim_line in elements['dimension_lines']:
                closest_drawing = self._find_closest_drawing(dim_line, elements['drawing_areas'])
                if closest_drawing:
                    measurements = self._extract_measurements(dim_line['doc'] or "")
                    link_info = {
                        'dimension_content': dim_line['doc'],
                        'dimension_spatial': dim_line['spatial'],
                        'linked_drawing': closest_drawing['meta'].get('image_path', ''),
                        'drawing_spatial': closest_drawing['spatial'],
                        'measurements': measurements,
                        'ocr_confidence': dim_line['meta'].get('ocr_confidence', 0),
                        'proximity_score': self._calculate_proximity(dim_line['spatial'], closest_drawing['spatial'])
                    }
                    spatial_links[sheet]['linked_dimensions'].append(link_info)
                    spatial_links[sheet]['measurements_found'].extend(measurements)

            for drawing in elements['drawing_areas']:
                spatial_links[sheet]['drawing_info'][drawing['meta'].get('image_path', '')] = {
                    'content': drawing['doc'],
                    'spatial': drawing['spatial'],
                    'ocr_confidence': drawing['meta'].get('ocr_confidence', 0)
                }
        return spatial_links
    
    def _parse_spatial_data(self, spatial_str: str) -> Dict[str, Any]:
        """Parse spatial JSON data from metadata"""
        try:
            return json.loads(spatial_str)
        except:
            return {}
    
    def _calculate_proximity(self, spatial1: Dict, spatial2: Dict) -> float:
        """Calculate spatial proximity between two elements"""
        try:
            bbox1 = spatial1.get('bbox_px', [0, 0, 0, 0])
            bbox2 = spatial2.get('bbox_px', [0, 0, 0, 0])
            
            # Calculate center points
            center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
            center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
            
            # Calculate distance
            distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
            
            # Normalize to 0-1 score (closer = higher score)
            max_distance = 1000  # Adjust based on typical drawing size
            proximity = max(0, 1 - (distance / max_distance))
            
            return proximity
            
        except:
            return 0.0
    
    def _find_closest_drawing(self, dim_line: Dict, drawing_areas: List[Dict]) -> Optional[Dict]:
        """Find the drawing area closest to a dimension line"""
        if not drawing_areas:
            return None
            
        best_drawing = None
        best_proximity = 0
        
        for drawing in drawing_areas:
            proximity = self._calculate_proximity(dim_line['spatial'], drawing['spatial'])
            if proximity > best_proximity:
                best_proximity = proximity
                best_drawing = drawing
        
        return best_drawing if best_proximity > 0.1 else None  # Threshold for valid links
    
    def _extract_measurements(self, text: str) -> List[str]:
        """Extract measurements from dimension line text"""
        if not text:
            return []
            
        measurements = []
        
        # Various measurement patterns
        patterns = [
            r'\b\d+[\'\u2032]\s*\d*[\"\u2033]?',  # Feet/inches: 12'6", 8'
            r'\b\d+\s*(?:in|inch|inches)\b',       # Inches: 24 in
            r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m)\b',   # Metric: 300mm, 3.5m
            r'\b\d+(?:\.\d+)?[\'\u2032]',         # Just feet: 12'
            r'\b\d+(?:\.\d+)?\"',                 # Just inches: 6"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            measurements.extend(matches)
        
        return list(set(measurements))  # Remove duplicates
    
    def get_sheet_summary(self, sheet_id: str) -> Dict[str, Any]:
        if self.text_collection is None:
            return {"error": "Text collection not available"}

        where_eq = {"sheet_id": sheet_id}
        pulled = self._fetch_all(self.text_collection, include=('metadatas','documents'), where=where_eq)
        docs = pulled.get('documents', [])
        metas = pulled.get('metadatas', [])

        elements_by_type, all_measurements = {}, []
        for doc, meta in zip(docs, metas):
            meta = meta or {}
            label = meta.get('label', 'Unknown')
            elements_by_type.setdefault(label, [])
            element_info = {
                'content': doc,
                'metadata': meta,
                'spatial': self._parse_spatial_data(meta.get('spatial', '{}')),
                'ocr_confidence': meta.get('ocr_confidence', 0)
            }
            elements_by_type[label].append(element_info)
            if label == 'dimension_line':
                all_measurements.extend(self._extract_measurements(doc or ""))

        # Links for this sheet — fetch only this sheet (all items)
        spatial_links = self.get_spatial_links(sheet_id=sheet_id)
        sheet_links = spatial_links.get(sheet_id, {})

        return {
            'sheet_id': sheet_id,
            'drawing_type': self._infer_drawing_type(sheet_id),
            'total_elements': len(docs),
            'elements_by_type': {k: len(v) for k, v in elements_by_type.items()},
            'all_measurements': sorted(set(all_measurements)),
            'dimension_drawing_links': len(sheet_links.get('linked_dimensions', [])),
            'detailed_elements': elements_by_type,
            'spatial_relationships': sheet_links
        }

    
    def find_measurements_in_context(self, measurement_query: str, context_keywords: List[str] = None) -> Dict[str, Any]:
        """
        Find measurements with their spatial context and linked drawings
        
        This is powerful for queries like "beam depth" or "room width"
        """
        # Search for dimension lines
        dim_results = self.text_collection.query(
            query_embeddings=self.encode_text_query(measurement_query).tolist(),
            n_results=50,  # Get more to filter later
            include=['metadatas', 'documents']
        )
        
        docs = dim_results.get('documents', [[]])[0]
        metas = dim_results.get('metadatas', [[]])[0]
        
        measurement_contexts = []
        
        for doc, meta in zip(docs, metas):
            # Filter to dimension lines only
            if meta.get('label') != 'dimension_line':
                continue
                
            measurements = self._extract_measurements(doc)
            
            if measurements:  # Only include if measurements found
                sheet_id = meta.get('sheet_id', '')
                
                # Get spatial context - find related drawing areas
                spatial_data = self._parse_spatial_data(meta.get('spatial', '{}'))
                
                # Get related drawing content using spatial proximity
                related_drawings = self._get_spatially_related_content(sheet_id, spatial_data)
                
                measurement_contexts.append({
                    'measurements': measurements,
                    'dimension_text': doc,
                    'sheet_id': sheet_id,
                    'ocr_confidence': meta.get('ocr_confidence', 0),
                    'spatial_location': spatial_data,
                    'related_drawings': related_drawings,
                    'context_score': self._score_measurement_context(doc, context_keywords or [])
                })
        
        # Sort by relevance (OCR confidence + context score)
        measurement_contexts.sort(
            key=lambda x: (x['ocr_confidence'] * 0.7 + x['context_score'] * 0.3), 
            reverse=True
        )
        
        return {
            'total_found': len(measurement_contexts),
            'measurement_contexts': measurement_contexts[:10],  # Top 10
            'unique_measurements': list(set([m for ctx in measurement_contexts for m in ctx['measurements']]))
        }
    
    def _get_spatially_related_content(self, sheet_id: str, target_spatial: Dict) -> List[Dict]:
        if not target_spatial or 'bbox_px' not in target_spatial:
            return []
        pulled = self._fetch_all(self.text_collection, include=('metadatas','documents'), where={"sheet_id": sheet_id})
        docs = pulled.get('documents', [])
        metas = pulled.get('metadatas', [])
        related = []
        for doc, meta in zip(docs, metas):
            meta = meta or {}
            if meta.get('label') == 'dimension_line':
                continue
            element_spatial = self._parse_spatial_data(meta.get('spatial', '{}'))
            proximity = self._calculate_proximity(target_spatial, element_spatial)
            if proximity > 0.3:
                related.append({
                    'content': doc, 'label': meta.get('label'),
                    'proximity_score': proximity,
                    'image_path': meta.get('image_path', ''),
                    'ocr_confidence': meta.get('ocr_confidence', 0)
                })
        related.sort(key=lambda x: x['proximity_score'], reverse=True)
        return related[:5]
    
    def _score_measurement_context(self, text: str, keywords: List[str]) -> float:
        """Score how relevant measurement context is to query keywords"""
        if not keywords or not text:
            return 0.0
            
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return min(1.0, matches / len(keywords))
    
    def get_drawing_by_name_or_type(self, search_term: str) -> Dict[str, Any]:
        """Find drawings by name, type, or content description"""
        
        # Search both collections for comprehensive results
        text_results = self.retrieve_text(search_term, k=15)
        image_results = self.retrieve_images_by_caption(search_term, k=10)
        
        # Group results by sheet
        sheets_found = {}
        
        for result_set in [text_results, image_results]:
            docs = result_set.get('documents', [[]])[0]
            metas = result_set.get('metadatas', [[]])[0]
            
            for doc, meta in zip(docs, metas):
                sheet_id = meta.get('sheet_id', 'Unknown')
                
                if sheet_id not in sheets_found:
                    sheets_found[sheet_id] = {
                        'sheet_id': sheet_id,
                        'drawing_type': self._infer_drawing_type(sheet_id),
                        'elements': [],
                        'total_confidence': 0,
                        'measurement_count': 0
                    }
                
                # Add element info
                measurements = self._extract_measurements(doc) if doc else []
                
                sheets_found[sheet_id]['elements'].append({
                    'content': doc,
                    'label': meta.get('label'),
                    'ocr_confidence': meta.get('ocr_confidence', 0),
                    'measurements': measurements
                })
                
                sheets_found[sheet_id]['total_confidence'] += meta.get('ocr_confidence', 0)
                sheets_found[sheet_id]['measurement_count'] += len(measurements)
        
        # Calculate average confidence and sort
        for sheet_info in sheets_found.values():
            if sheet_info['elements']:
                sheet_info['avg_confidence'] = sheet_info['total_confidence'] / len(sheet_info['elements'])
            else:
                sheet_info['avg_confidence'] = 0
        
        # Sort by relevance (confidence + element count)
        sorted_sheets = sorted(
            sheets_found.values(),
            key=lambda x: (x['avg_confidence'] * 0.6 + len(x['elements']) * 0.4),
            reverse=True
        )
        
        return {
            'query': search_term,
            'sheets_found': len(sorted_sheets),
            'results': sorted_sheets[:10],  # Top 10 most relevant sheets
            'total_elements': sum(len(s['elements']) for s in sorted_sheets),
            'total_measurements': sum(s['measurement_count'] for s in sorted_sheets)
        }

    def get_sqlite_tables(self):
        """Direct SQLite inspection for debugging"""
        sqlite_path = os.path.join(self.vectordb_path, "chroma.sqlite3")
        
        if not os.path.exists(sqlite_path):
            return {"error": "SQLite file not found"}
            
        try:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            info = {"tables": tables}
            
            # Get collections info
            if 'collections' in tables:
                cursor.execute("SELECT name, id FROM collections;")
                collections = cursor.fetchall()
                info['collections'] = collections
                
            conn.close()
            return info
            
        except Exception as e:
            return {"error": f"SQLite inspection failed: {e}"}

if __name__ == "__main__":
    # Test the enhanced system with document analysis and spatial linking
    rag = ArchRAGSystem()
    
    print("=== 📊 DRAWING INVENTORY ===")
    inventory = rag.get_drawing_inventory()
    print(f"Total sheets: {inventory['total_sheets']}")
    print(f"Total elements: {inventory['total_elements']}")
    print(f"Drawing types: {json.dumps(inventory['drawing_types'], indent=2)}")
    print(f"Sheets with dimensions: {inventory['summary']['sheets_with_dimensions']}")
    print(f"Most common elements: {inventory['summary']['most_common_labels'][:5]}")
    
    # Get a sample sheet for detailed analysis
    sample_sheets = list(inventory['sheets_detail'].keys())[:3]
    
    for sheet_id in sample_sheets:
        print(f"\n=== 📋 SHEET ANALYSIS: {sheet_id} ===")
        sheet_summary = rag.get_sheet_summary(sheet_id)
        print(f"Drawing type: {sheet_summary['drawing_type']}")
        print(f"Elements: {sheet_summary['elements_by_type']}")
        print(f"Measurements found: {len(sheet_summary['all_measurements'])}")
        print(f"Dimension-drawing links: {sheet_summary['dimension_drawing_links']}")
        
        if sheet_summary['all_measurements']:
            print(f"Sample measurements: {sheet_summary['all_measurements'][:5]}")
    
    print(f"\n=== 🔍 SPATIAL MEASUREMENT SEARCH ===")
    
    # Test measurement search with spatial context
    measurement_search = rag.find_measurements_in_context(
        "beam depth clearance", 
        context_keywords=["beam", "structural", "depth"]
    )
    
    print(f"Found {measurement_search['total_found']} measurement contexts")
    print(f"Unique measurements: {measurement_search['unique_measurements'][:10]}")
    
    # Show top measurement contexts with spatial links
    for i, ctx in enumerate(measurement_search['measurement_contexts'][:3]):
        print(f"\n{i+1}. Sheet: {ctx['sheet_id']} | OCR: {ctx['ocr_confidence']:.2f}")
        print(f"   Measurements: {ctx['measurements']}")
        print(f"   Context: {ctx['dimension_text'][:100]}...")
        print(f"   Related drawings: {len(ctx['related_drawings'])}")
        
        for related in ctx['related_drawings'][:2]:
            print(f"     → {related['label']} (proximity: {related['proximity_score']:.2f})")
    
    print(f"\n=== 🎯 DRAWING SEARCH BY TYPE ===")
    
    # Test drawing search by type
    structural_drawings = rag.get_drawing_by_name_or_type("structural steel details")
    print(f"Found {structural_drawings['sheets_found']} sheets for 'structural steel details'")
    
    for result in structural_drawings['results'][:3]:
        print(f"📄 {result['sheet_id']} ({result['drawing_type']})")
        print(f"   Elements: {len(result['elements'])}, Measurements: {result['measurement_count']}")
        print(f"   Avg confidence: {result['avg_confidence']:.2f}")
    
    print("\n✅ Enhanced RAG system testing complete!")