import itertools
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from contriever_functions import retrieve_similarities_dot, retrieve_similarities_cos, get_sent_embeddings

class CHECK():
    def __init__(self, 
                 model, 
                 tokenizer, 
                 refined_model,
                 type_prompt='', 
                 extraction_prompt='', 
                 subq_prompt='', 
                 qa_prompt='', 
                 type_template='',
                 is_vicuna=False, 
                 similarity='cos',
                 sim_thresh=0.8,
                 do_type_check=True,
                 do_temp_increase=True,
                 device='cpu'):
        self.model = model
        self.refined = refined_model
        self.tokenizer = tokenizer
        self.te_prompt = type_prompt
        self.extraction_prompt = extraction_prompt
        self.subq_prompt = subq_prompt
        self.qa_prompt = qa_prompt
        self.is_vicuna = is_vicuna
        self.similarity = similarity
        self.sim_thresh = sim_thresh
        self.device = device
        
        self.edit_bank = []
        self.embedding_bank = []
        self.edited_entity_bank = []
        
        self.contriever = AutoModel.from_pretrained('facebook/contriever-msmarco').to(device)
        self.c_tok = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')

        self.template_bank = {}
        all_r = []
        for r in type_template:
            if r.endswith('\n'):
                r = r.replace('\n', '')
            r = r.split(': ')
            r_current = r[0]
            all_r.append(r_current)
            types = r[1].split(', ')
            self.template_bank[r_current] = {'in': types[0].split(' '), 'out': types[1].split(' ')}
        self.template_embedding = get_sent_embeddings(all_r, self.contriever, self.c_tok, self.device)

        self.ro_extractor = model
        self.ro_tokenizer = tokenizer

    def get_entity(self, text):
        '''
        Extract the entity of an input with the Refined entity linking model.
        '''
        final_entity = ''
        entity = self.refined.process_text(text)
        if len(entity) > 0:
            # The initially extracted entity
            entity = entity[-1]
            final_entity = entity.text # initial entity
            # See if entity in the input was linking to an entity and use that instead
            if not isinstance(entity.predicted_entity, type(None)): 
                if not isinstance(entity.predicted_entity.wikipedia_entity_title, type(None)):
                    final_entity = entity.predicted_entity.wikipedia_entity_title # linked entity
        return final_entity
    
    def get_type(self, subject, question, num_new_tokens=50):
        '''
        Get entity type.

        This function is specific to the Refined entity linking model.
        Other named entity recognition or entity linking models can be used.
        The function will need to be rewritten to extract person, places, and things
        with the desired model, though. 
        '''
        # Try to get the entity type with the Refined model.
        the_type = ''
        entity = self.refined.process_text(question)
        if len(entity) > 0:
            entity = entity[-1]
            if entity.coarse_mention_type == 'PERSON':
                return 'person'
            elif entity.coarse_mention_type in ['FAC', 'GPE', 'LOC']:
                return 'place'
            elif entity.coarse_mention_type in ['NORP', 'ORG', 'PRODUCT', 'WORK_OF_ART', 'LANGUAGE', 'LAW', 'EVENT']:
                return 'thing'
 
        # If Refined failed to return an entity type, ask the LLM to do it. 
        type_extraction = f'{self.te_prompt}\nEntity: {subject}\nType:'
        the_type = self.query_model(self.ro_extractor, 
                                    self.ro_tokenizer, 
                                    type_extraction, 
                                    num_new_tokens, 
                                    stop_strings=['\n'], 
                                    model_is_vicuna=self.is_vicuna)
        if the_type.endswith(' ') or the_type.endswith('\n'):
            the_type = the_type[:-1]
        return the_type
    
    def add_edits(self, requested_edits, edit_sentences):
        '''
        Insert edits into the CHECK embedding memory.
        Edits expected to be in (s, r, o) form
        '''
        self.edit_bank = requested_edits

        # Embed the subject and relationship
        sr = [' '.join(requested_edits[i][:2]) for i in range(len(requested_edits))]
        self.embedding_bank = get_sent_embeddings(sr, self.contriever, self.c_tok, self.device)

        # Keep the edit objects in their own list. Indexes between the embedding and object lists match.
        self.edited_entity_bank = []
        for sentence in edit_sentences:
            final_entity = self.get_entity(sentence)
            self.edited_entity_bank.append(final_entity)
    
    def query_model(self, model, tokenizer, input_text, num_new_tokens, stop_strings=['Triple: (', 'END'], model_is_vicuna=False, temperature=0.001):
        '''
        Pass an input to the LLM.
        '''
        tokens = tokenizer(input_text, padding=False, truncation=True, return_tensors='pt').to(self.device)
    
        # Made to avoid inputs that will cause the GPU to run out of memory.
        if tokens.input_ids.shape[1] > 2048-num_new_tokens-1:
            return ''
        
        generated_tokens = model.generate(tokens.input_ids, 
                                          attention_mask=tokens.attention_mask,
                                          do_sample=True,
                                          temperature=temperature,
                                          max_new_tokens=num_new_tokens,
                                          stop_strings=stop_strings,
                                          tokenizer=tokenizer)
                                        
        generated_text = tokenizer.decode(generated_tokens[0])[len(input_text):]
        if model_is_vicuna: # Vicuna adds a set of sentence start tokens
            generated_text = generated_text[5:]
        if generated_text.startswith(' '):
            generated_text = generated_text[1:]
        if '</s>' in generated_text:
            generated_text = generated_text.replace('</s>', '')
        if '<|endoftext|>' in generated_text:
            generated_text = generated_text.replace('<|endoftext|>', '')
        return generated_text

    def check_sr(self, s, r):
        '''
        Find an embedded edit subject, relationship pair close to the current pair, if one exists.
        '''
        closest_o = None
        closest_val = 0
        closest_sro = None
        
        # Try to link entity
        try:
            current_subject = self.refined.process_text(s)
        except:
            current_subject = []
        cs = ''
        if len(current_subject) > 0:
            current_subject = current_subject[-1]
            cs = current_subject.text
            if not isinstance(current_subject.predicted_entity, type(None)): 
                if not isinstance(current_subject.predicted_entity.wikipedia_entity_title, type(None)):
                    cs = current_subject.predicted_entity.wikipedia_entity_title

        # Embed current entity and relationship
        sro_embed = get_sent_embeddings([' '.join([s, r])], self.contriever, self.c_tok, self.device)
        
        found_similar = False

        # Look for the most similar embedding
        for i in range(len(self.edit_bank)):
            # Check to see if the current entity is directly in the edit bank for filtering
            current_similar = False
            c = self.edited_entity_bank[i]
            if cs == c and cs != '' and c != '':
                current_similar = True
                if not found_similar:
                    found_similar = True
                    closest_val = 0

            # Compare embeddings with cosine similarity or dot product
            if self.similarity == 'dot':
                similarity = retrieve_similarities_dot(sro_embed, self.embedding_bank[i].unsqueeze(0), device=self.device)[0][0].item()
            elif self.similarity == 'cos':
                similarity = retrieve_similarities_cos(sro_embed, self.embedding_bank[i].unsqueeze(0), device=self.device)[0].item()
            if (similarity > closest_val and not found_similar) or (similarity > closest_val and found_similar and current_similar):
                    closest_val = similarity
                    closest_o = self.edit_bank[i][-1]
                    closest_sro = self.edit_bank[i]

        # Use the highest similarity embedding if the similarity score meets a threshold
        if closest_val > self.sim_thresh:
            return closest_o
        return None

    def answer_question(self, question, num_new_tokens=50):
        '''
        Multi-hop question answering with CHECK.
        '''
        # The input prompt: extraction prompt + MHQ
        rxo_extraction = f'{self.extraction_prompt}\n\nQuestion: {question}\nSRO:'

        best_chain = None
        best_in_out = None
        total_least_moves = None
        total_least_conflicts = None

        ro = None
        
        # Get the question entity and its type
        init_s = self.get_entity(question)
        init_s_type = self.get_type(init_s, question, num_new_tokens)

        for temperature in [0.00001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            # Extract the relationship chain from the multi-hop question
            ro = self.query_model(self.model, 
                                  self.tokenizer, 
                                  rxo_extraction, 
                                  num_new_tokens, 
                                  stop_strings=['\n'], 
                                  model_is_vicuna=self.is_vicuna,
                                  temperature=temperature) 

            # Formatting
            ro = ro.split('\n')[0]

            s = init_s
            s_type = init_s_type
            
            ro = ro[:-1].split(' | ')
            if ro[0].startswith('| '):
                ro[0] = ro[0][2:]
            ro[-1] = ro[-1][:-1]

            if s in ro:
                ro.remove(s)

            if len(ro) < 1:
                continue
            elif len(ro) == 1 and temperature < 0.7:
                continue
            elif len(ro) < 1:
                return None
        
            # Embed each relationship in the chain
            ro_embeddings = get_sent_embeddings(ro, self.contriever, self.c_tok, self.device)

            # Get the types of the relationships
            in_outs = []
            new_ro = []
            for i in range(ro_embeddings.shape[0]):
                highest_val = 0
                highest_embed = None
                highest_r = None

                for j in range(self.template_embedding.shape[0]):
                    if self.similarity == 'dot':
                        similarity = retrieve_similarities_dot(ro_embeddings[i].unsqueeze(0), self.template_embedding[j].unsqueeze(0), device=self.device)[0][0].item()
                    elif self.similarity == 'cos':
                        similarity = retrieve_similarities_cos(ro_embeddings[i].unsqueeze(0), self.template_embedding[j].unsqueeze(0), device=self.device)[0].item()

                    if similarity > highest_val:
                        highest_val = similarity
                        highest_embed = self.template_bank[list(self.template_bank)[j]]
                        if similarity > self.sim_thresh:      
                            highest_r = list(self.template_bank)[j]    
                        else:
                            highest_r = ro[i]
        
                new_ro.append(highest_r)
                in_outs.append(highest_embed)

            ro = new_ro

            # Add the subject to the relationship chain if it has a type
            if s_type != '':
                ro.append(s)
                in_outs.append({'in': '', 'out': [s_type]})
            permute_tracker = list(range(len(in_outs)))

            # Count the number of conflicts in the relationship chain
            conflicts = [True] * (len(in_outs)-1)
            for i in reversed(range(len(in_outs))):
                if i == 0:
                    break
                if set(in_outs[i]['out']).intersection(set(in_outs[i-1]['in'])):
                    conflicts[i-1] = False

            # Do relationship chain repair
            least_num_conflicts = 100
            least_num_moves = 100
            chosen_idx = -1
            has_conflicts = True in conflicts
            if has_conflicts:
                inout_permutations = list(itertools.permutations(in_outs))
                rc_permutations = list(itertools.permutations(ro))
                permute_track_list = list(itertools.permutations(permute_tracker))

                for i, path in enumerate(inout_permutations):
                    current_conflicts = 0
                    move_count = -1

                    for j in reversed(range(len(path))):
                        if j == 0: 
                            break

                        if not set(path[j]['out']).intersection(set(path[j-1]['in'])):
                            current_conflicts += 1

                    for j in range(len(path)):
                        if permute_tracker[j] != permute_track_list[i][j]:
                            move_count += 1

                    if (current_conflicts <= least_num_conflicts and move_count < least_num_moves):# or current_conflicts < least_num_conflicts:
                        least_num_conflicts = current_conflicts
                        least_num_moves = move_count
                        chosen_idx = i

                if chosen_idx > -1:
                    if s_type != '':
                        ro = rc_permutations[chosen_idx][:-1]
                        in_outs = inout_permutations[chosen_idx][:-1]
                    else:
                        ro = rc_permutations[chosen_idx]
                        in_outs = inout_permutations[chosen_idx]
                    if least_num_conflicts < 0.5:
                        has_conflicts = False
            
            elif s_type != '':  
                ro = ro[:-1]
                in_outs = in_outs[:-1]

            if not has_conflicts:
                best_chain = ro
                best_in_out = in_outs
                break
            elif isinstance(best_chain, type(None)):
                best_chain = ro
                best_in_out = in_outs
                total_least_moves = least_num_moves
                total_least_conflicts = least_num_conflicts
            elif (least_num_conflicts <= total_least_conflicts and least_num_moves < total_least_moves):# or least_num_conflicts < total_least_conflicts:
                best_chain = ro
                best_in_out = in_outs
                total_least_moves = least_num_moves
                total_least_conflicts = least_num_conflicts
            
            if temperature > 0.95:
                ro = best_chain
                in_outs = best_in_out

        # Start the chain traversal
        for i, r in enumerate(reversed(ro)):
            # Look for similar edit subject, relationship
            o = self.check_sr(s, r)
            # If an edit is found, nothing additional needs to be done
            # If no similar edit is found, query the LLM for an object
            if o is None:
                # Rephrase the subject, relationship into a question
                sr_triple = f'Triple: | {s} | {r} |'
                new_subq = self.query_model(self.ro_extractor, 
                                            self.ro_tokenizer, 
                                            f'{self.subq_prompt}\n\nRephrase the following triple as a question:\n{sr_triple}\nQuestion:', 
                                            num_new_tokens, 
                                            stop_strings=['?'], 
                                            model_is_vicuna=self.is_vicuna)
                new_subq = new_subq.split('\n')[0].split(':')[-1]
                if new_subq.startswith(' '):
                    new_subq = new_subq[1:]
                # Answer the question
                o = self.query_model(self.model, 
                                        self.tokenizer, 
                                        f'{self.qa_prompt}\n\nQuestion:{new_subq}\nAnswer:', 
                                        num_new_tokens, 
                                        stop_strings=['|'],
                                        model_is_vicuna=self.is_vicuna)
                if o.startswith(' '):
                    o = o[1:]
                if o.endswith('|'):
                    o = o[:-1]

            # Make the current object the next subject to be paired with the next relationship
            s = o

        return s