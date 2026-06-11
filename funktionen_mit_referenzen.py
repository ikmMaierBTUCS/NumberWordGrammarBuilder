import numpy as np
from num2words import num2words
import itertools
import bisect
from itertools import combinations
from itertools import permutations
from sympy import Matrix
from diophantine import solve
import pandas as pd
from alphabet_detector import AlphabetDetector
import types
import matplotlib.pyplot as plt
import random
import os

import unicodedata

class StopLearning(Exception):
    pass

class UndoLearning(Exception):
    pass

# Replaceable input function for hybrid2 mode (set by GUI)
_hybrid2_input_func = None

class Highlight:
    def __init__(self, voc, start):
        self.number=voc.number
        self.numeral=voc.word
        self.start=start
        self.root=voc.word
        self.mapping=[voc.number]
        self.word=voc.word
        self.key=voc.key if hasattr(voc, 'key') else voc.word
    def end(self):
        return self.start+len(self.numeral)
    def hlrange(self):
        return range(self.start+1,self.end())
    def voc(self):
        return Vocabulary(self.number,self.numeral,key=self.key)
    
def to_scf(eingabe, kenntnis_lexikon=None):
    lexikon = kenntnis_lexikon
    if lexikon is None:
        lexikon = globals().get('kenntnis_lexikon', None)
    if type(eingabe) is str:
        if isinstance(lexikon, dict) and eingabe in lexikon.keys():
            return lexikon[eingabe]
        else:
            print(eingabe)
            print(lexikon.keys())
            raise KeyError("String input has to be a key in kenntnis_lexikon")
    elif type(eingabe) is SCFunction:
        return eingabe
    elif type(eingabe) is Vocabulary:
        return SCFunction(eingabe.word,[],[eingabe.number],key=eingabe.key if hasattr(eingabe, 'key') else eingabe.word, kenntnis_lexikon=lexikon)
    else:
        raise TypeError("Input has to be a Vocabulary or an SCFunction")

def to_key(eingabe, kenntnis_lexikon=None, fallback=True):
    lexikon = kenntnis_lexikon
    if lexikon is None:
        lexikon = globals().get('kenntnis_lexikon', None)
    if type(eingabe) is str:
        if isinstance(lexikon, dict) and eingabe in lexikon.keys():
            return eingabe
        else:
            raise KeyError("String input has to be a key in kenntnis_lexikon")
    elif type(eingabe) is SCFunction or type(eingabe) is Vocabulary:
        if not fallback or not isinstance(lexikon, dict) or eingabe.key in lexikon:
            return eingabe.key
        # eingabe.key existiert nicht mehr im Lexikon.
        # Suche einen Key, dessen Eintrag alle confirmed Outputs überdeckt.
        eingabe_outputs = {(round(o.mapping[-1]), o.root) for o in eingabe.all_outputs(only_confirmed=True)}
        for k, v in lexikon.items():
            v_outputs = {(round(o.mapping[-1]), o.root) for o in v.all_outputs(only_confirmed=True)}
            if eingabe_outputs.issubset(v_outputs):
                return k
        return eingabe.key
    else:
        raise TypeError("Input has to be a Vocabulary or an SCFunction")
    
class SCFunction:
    '''
    Root is the exponent of the function where _ mark input slots
    Inputrange is a list of lists. The nth lists all SCFunctions that may enter the nth input slot
    Mapping is a list of coefficients. The nth coefficient is the factor by which the nth input would have to be multiplied
        Exception: the last coefficient is the constant coefficient.
    '''
    def __init__(self,root,i,mapping,c=[],key=None, kenntnis_lexikon=None):
        if not type(root) is str:
            raise TypeError("Root has to be a string")
        if not type(i) is list:
            raise TypeError("Inputrange has to be a list")
        dimension=root.count('_')
        if not len(i) == dimension:
            print(root)
            print(i)
            raise TypeError(str(dimension)+"-dimensional function needs "+str(dimension)+" component domains.")
        self.inputrange = []
        for comp in range(len(i)):
            component = i[comp]
            if not type(component) is list:
                print(type(component))
                raise TypeError("All component domains have to be lists")
            if len(component) == 0:
                raise ValueError("Component domains cannot be empty")
            elif len(component) == 1:
                entry = component[0]
                if type(entry) is SCFunction and entry.dimension() > 0:
                    print('WARNING: singleton component domain contains a non-atomic SCFunction.')
                    entry.present 
                    print('in')                         
                    print(root)
                    print('This is rarely supposed to occur rightfully.')
                    i[comp] = [to_key(entry, kenntnis_lexikon=kenntnis_lexikon)]
            elif len(component) > 1:
                for entry in component:
                    if not type(entry) in [str,Vocabulary, SCFunction]:
                        print(type(entry))
                        raise TypeError("All entries of input component domains have to be presentable as lexicon keys")
                original_component = component
                component = [to_key(entry, kenntnis_lexikon=kenntnis_lexikon) for entry in component]
                component = list(set(component))
                if len(component) == 1:
                    # Dedup reduced multi-entry to singleton — preserve original type instead of str key
                    component = [next(e for e in original_component if to_key(e, kenntnis_lexikon=kenntnis_lexikon) == component[0])]
                    i[comp] = component
            self.inputrange += [list(set(component))] # deduplicate component domain
        

        if not type(mapping) is list or dimension+1!=len(mapping):
            print(type(mapping))
            try:
                print(len(mapping))
            except:
                pass
            raise TypeError("Mapping has to be a list of "+str(dimension)+"+1 coefficients")
        #for coeff in mapping:
            #if not type(coeff) is int and not type(coeff) is float:
                #print(type(coeff))
                #raise TypeError("All coefficients of the mapping have to be integers or floats")

        if c == []:
            c = i
        else:
            # Validate confirmed_inputrange format
            if not type(c) is list:
                raise TypeError("Confirmed inputrange has to be a list")
            if not len(c) == dimension:
                raise TypeError(str(dimension)+"-dimensional function needs "+str(dimension)+" confirmed component domains.")
            for comp in range(len(c)):
                component = c[comp]
                if not type(component) is list:
                    print(type(component))
                    raise TypeError("All confirmed component domains have to be lists")
                if len(component) == 0:
                    raise ValueError("Component domains cannot be empty")
                else:
                    inputrange_keys = set(to_key(e, kenntnis_lexikon=kenntnis_lexikon) for e in i[comp])
                    for entry in component:
                        if to_key(entry, kenntnis_lexikon=kenntnis_lexikon) not in inputrange_keys:
                            print("Confirmed inputrange has to be a subset of inputrange. Problem at component "+str(comp)+":")
                            print("Inputrange component:", i[comp])
                            print("Confirmed inputrange component:", component)
                            raise ValueError("Confirmed inputrange has to be a subset of inputrange.")
                    if len(component) > 1:
                        for entry in component:
                            if not type(entry) in [str,Vocabulary, SCFunction]:
                                print(type(entry))
                                raise TypeError("All entries of input component domains have to be representable as lexicon keys")
                        original_c_component = component
                        c[comp] = [to_key(entry, kenntnis_lexikon=kenntnis_lexikon) for entry in component]
                        c[comp] = list(set(c[comp]))
                        if len(c[comp]) == 1:
                            c[comp] = [next(e for e in original_c_component if to_key(e, kenntnis_lexikon=kenntnis_lexikon) == c[comp][0])]
                c[comp] = list(set(c[comp])) # deduplicate confirmed component domain
        self.confirmed_inputrange = c 

        self.root = root
        self.mapping = mapping
        self.kenntnis_lexikon = kenntnis_lexikon
        if isinstance(kenntnis_lexikon, dict):
            globals()['kenntnis_lexikon'] = kenntnis_lexikon

        # Berechne maximalen und minimalen Wert
        maxi = self.mapping[-1]
        mini = self.mapping[-1]
        conf_maxi = self.mapping[-1]
        conf_mini = self.mapping[-1]
        conf_max_input = 0
        max_input = 0
        for i in range(len(self.mapping)-1):
            conf_max_input = max(conf_max_input, max(to_scf(e, kenntnis_lexikon=kenntnis_lexikon).maximum for e in self.confirmed_inputrange[i]))
            max_input = max(max_input, max(to_scf(e, kenntnis_lexikon=kenntnis_lexikon).maximum for e in self.inputrange[i]))
            if self.mapping[i] >= 0:
                maxi += self.mapping[i] * max([to_scf(e, kenntnis_lexikon=kenntnis_lexikon).maximum for e in self.inputrange[i]])
                mini += self.mapping[i] * min([to_scf(e, kenntnis_lexikon=kenntnis_lexikon).minimum for e in self.inputrange[i]])
                conf_maxi += self.mapping[i] * max([to_scf(e, kenntnis_lexikon=kenntnis_lexikon).maximum for e in self.confirmed_inputrange[i]])
                conf_mini += self.mapping[i] * min([to_scf(e, kenntnis_lexikon=kenntnis_lexikon).minimum for e in self.confirmed_inputrange[i]])
            else:
                maxi += self.mapping[i] * min([to_scf(e, kenntnis_lexikon=kenntnis_lexikon).minimum for e in self.inputrange[i]])
                mini += self.mapping[i] * max([to_scf(e, kenntnis_lexikon=kenntnis_lexikon).maximum for e in self.inputrange[i]])
                conf_maxi += self.mapping[i] * min([to_scf(e, kenntnis_lexikon=kenntnis_lexikon).minimum for e in self.confirmed_inputrange[i]])
                conf_mini += self.mapping[i] * max([to_scf(e, kenntnis_lexikon=kenntnis_lexikon).maximum for e in self.confirmed_inputrange[i]])
        self.maximum = round(maxi)
        self.minimum = round(mini)
        self.confirmed_maximum = round(conf_maxi)
        self.confirmed_minimum = round(conf_mini)
        self.maximal_input = round(max_input)
        self.confirmed_maximal_input = round(conf_max_input)

        if key is None:
            key_ending = ['']+list(range(100))
            for end in key_ending:
                key = self.root + str(end)
                if kenntnis_lexikon is not None and not key in kenntnis_lexikon.keys():
                    self.key = key
                    break
            if not hasattr(self, 'key'):
                self.key = self.root
        else:
            self.key = key

    def __eq__(self,other):
        if isinstance(self,other.__class__):
            if self.root != other.root or self.mapping != other.mapping:
                return False
            else:
                for comp in range(self.dimension()):
                    r_m = set((to_scf(e).root,tuple(to_scf(e).mapping)) for e in self.inputrange[comp])
                    other_r_m = set((to_scf(e).root,tuple(to_scf(e).mapping)) for e in other.inputrange[comp])
                    if r_m != other_r_m:
                        return False
                return True
        else:
            return NotImplemented
            
    def __hash__(self):
        return hash((self.root,tuple(self.mapping)))
        
    def dimension(self):
        '''
        Dimension = Number of input slots
        '''
        return self.root.count('_')
    def number_inputs(self, only_confirmed = False):
        ni = []
        if only_confirmed:
            ir = self.confirmed_inputrange
        else:
            ir = self.inputrange
        for comp in ir:
            ni += [[to_scf(entr).sample().mapping[-1] for entr in comp]]
        return ni
    def input_numberbase(self):
        build_base=[]
        #print(len(self.inputrange))
        base_complete = False
        number_of_variable_input_slots = len([comp for comp in range(self.dimension()) if len(self.inputrange[comp]) > 1])
        for root_inputx in cartesian_product(self.confirmed_inputrange):
            for final_inputx in cartesian_product([to_scf(entr).all_outputs() for entr in root_inputx]):
                #print(build_base)
                #print([component.number for component in final_inputx]+[1])
                #if not inputx in span(build_base): # matrank(buildbase+inputx)=len(Buildbase)+1
                if np.linalg.matrix_rank(np.array(build_base+[[to_scf(component).mapping[-1] for component in final_inputx]+[1]], dtype=np.float64),tol=None)==len(build_base)+1:
                    #print(str(self.insert(inputx).number)+' is linear independent')
                    build_base=build_base+[[to_scf(component).mapping[-1] for component in final_inputx]+[1]]
                    #print([self.insert(inputx).number])
                if len(build_base) == number_of_variable_input_slots + 1:
                    #print('base complete')
                    base_complete = True
                    break
            if base_complete:
                break
        return build_base
        
    def actual_dimension(self):
        '''
        1 + Dimension of input range with respect to affine linearity
        '''
        numbers = [[ou.mapping[-1] for j in range(len(self.inputrange[i])) for ou in to_scf(self.inputrange[i][j]).sample_outputs(size=2, only_confirmed=True)] for i in range(self.dimension())]
        return np.linalg.matrix_rank(np.array(cartesian_product(numbers+[[1]]), dtype=np.float64))
    def insert(self,inputx):
        '''
        Requires a dimension-long list of input SCFunctions
        Return a new SCFunction where all input SCFunctions are inserted in their respective slot
        Updates inputrange and mapping with respect to new inputslots originating from the input SCFunctions
        '''
        
        # catch errors
        if not type(inputx) is list:
            print(type(inputx))
            raise TypeError('Input has to be a list')
        if len(self.inputrange) != len(inputx): #or len(self.inputrange) == 0:
            raise TypeError("Input does not match dimension.") # or "+self.root+" has no inputslots.")
        for component in inputx:
            if not type(component) is Vocabulary and not type(component) is SCFunction:
                print(type(component))
                raise TypeError('All components of the input have to be a Vocabulary or an SCFunction')
                
        # trouble shoot if input is not in the inputrange
        for entry in range(len(inputx)):
            if inputx[entry].root not in [to_scf(inputfunction).root for inputfunction in self.inputrange[entry]]:
                #print("Input "+str([inp.root for inp in inputx])+" is not in the input range of "+str(self.root))
                break
                
        # initialize root, inputrange and mapping of composed SCF
        rootparts = self.root.split('_')
        output_root = ''
        output_i = []
        output_c = []
        output_mapping = []
        constant_coefficient = self.mapping[-1]
        key = self.key
        
        # extend root, inputrange and mapping
        for inp in range(len(inputx)):
            output_root += rootparts[inp] + inputx[inp].root
            output_i += inputx[inp].inputrange
            output_c += inputx[inp].confirmed_inputrange
            output_mapping += [self.mapping[inp] * coeff for coeff in inputx[inp].mapping[:-1]]
            constant_coefficient += self.mapping[inp] * inputx[inp].mapping[-1]
            
        # finish root and mapping and return composed SCF
        output_root += rootparts[-1]
        output_mapping += [constant_coefficient]
        return SCFunction(output_root,output_i,output_mapping,output_c,key=key, kenntnis_lexikon=self.kenntnis_lexikon)

        
    def sample(self, size = -1, only_confirmed = False):
        '''
        If not size given, returns one output of self
        If size given, returns a random size-length list of outputs of self
        '''
        if self.dimension() == 0:
            return self
        else:
            if only_confirmed:
                ir = self.confirmed_inputrange
            else:
                ir = self.inputrange
            if size == -1:
                return self.insert([to_scf(comp[0]).sample() for comp in ir])
            else:
                inputvectors = cartesian_product(ir)
                size = min([size, len(inputvectors)])
                inputsample = random.sample(inputvectors,size)
                return [self.insert([to_scf(e) for e in i]) for i in inputsample]

    def random_output(self, only_confirmed = False, component_ignores=None):
        '''
        Return one random output without building the full cartesian product.
        Mirrors all_outputs exactly: same base case, same filtering, same recursion via insert.
        Returns None if no valid output can be produced (dead branch due to ignore filtering).
        component_ignores: list of lists, one ignore-list per component (per-component cycle prevention).
        '''
        # Same base case as all_outputs
        if self.dimension() == 0:
            return SCFunction(self.root, [], [round(self.mapping[0])],key=self.key, kenntnis_lexikon=self.kenntnis_lexikon)
        if only_confirmed:
            ir = self.confirmed_inputrange
        else:
            ir = self.inputrange
        if component_ignores is None:
            component_ignores = [[] for _ in range(self.dimension())]
        # Filter each component using its own ignore list + self.key
        ir_filtered = []
        lex = self.kenntnis_lexikon
        for comp_idx, comp in enumerate(ir):
            comp_ignore = component_ignores[comp_idx]
            if self.key not in comp_ignore:
                comp_ignore = comp_ignore + [self.key]
            filtered_comp = []
            for f in comp:
                if isinstance(f, str):
                    f_key = f
                    if isinstance(lex, dict) and f in lex:
                        resolved_key = lex[f].key
                    else:
                        resolved_key = f
                else:
                    f_key = to_scf(f, kenntnis_lexikon=lex).key
                    resolved_key = f_key
                if f_key not in comp_ignore and resolved_key not in comp_ignore:
                    filtered_comp.append(f)
            ir_filtered.append(filtered_comp)
        # Pick one random input vector (same structure as cartesian_product but just one sample)
        inputvector = []
        orig_entries = []
        for comp in ir_filtered:
            if not comp:
                return None  # empty component = dead branch (cartesian_product would yield nothing)
            chosen = random.choice(comp)
            orig_entries.append(chosen)
            inputvector.append(to_scf(chosen, kenntnis_lexikon=lex))
        # Build per-component ignore lists for the inserted function.
        # After insert, the new components come from inp_0's sub-components, then inp_1's, etc.
        # Each sub-component of inp_i inherits comp_i's ignore list + self.key + key(inp_i).
        next_component_ignores = []
        for comp_idx, iv in enumerate(inputvector):
            iv_key = to_key(iv, kenntnis_lexikon=lex, fallback=False)
            orig = orig_entries[comp_idx]
            orig_key = orig if isinstance(orig, str) else to_key(orig, kenntnis_lexikon=lex, fallback=False)
            base_ignore = component_ignores[comp_idx] + [self.key, iv_key]
            if orig_key not in base_ignore:
                base_ignore = base_ignore + [orig_key]
            for _ in range(iv.dimension()):
                next_component_ignores.append(base_ignore)
        return self.insert(inputvector).random_output(only_confirmed=only_confirmed, component_ignores=next_component_ignores)

    def sample_outputs(self, size = 30, only_confirmed = False):
        '''
        Return a random sample of outputs without enumerating all of them.
        Uses per-component ignore lists to prevent infinite recursion through self-referencing SCFunctions.
        '''
        if size <= 0:
            return []
        # Initialize per-component ignore lists, each containing self.key
        component_ignores = [[self.key] for _ in range(self.dimension())]
        results = []
        seen = set()
        max_attempts = size * 3  # Prevent infinite loops if most branches are dead
        attempts = 0
        while len(results) < size and attempts < max_attempts:
            r = self.random_output(only_confirmed, component_ignores=component_ignores)
            if r is not None and r.root not in seen:
                results.append(r)
                seen.add(r.root)
            attempts += 1
        return results
    
    def all_outputs(self, only_confirmed = False, component_ignores=None, _depth=0):
        '''
        return all final SCFunctions (vocabulary) without unsatisfied '_'s left, that are derivable from 
        component_ignores: list of lists, one ignore-list per component (per-component cycle prevention).
        '''
        #print('alloutputs of '+self.root)
        if self.dimension() == 0:
            return [SCFunction(self.root, [], [round(self.mapping[0])], key=self.key, kenntnis_lexikon=self.kenntnis_lexikon)]
        else:
            if only_confirmed:
                ir = self.confirmed_inputrange
            else:
                ir = self.inputrange
            if component_ignores is None:
                component_ignores = [[] for _ in range(self.dimension())]
            # Filter each component using its own ignore list + self.key
            ir_filtered = []
            lex = self.kenntnis_lexikon
            for comp_idx, comp in enumerate(ir):
                comp_ignore = component_ignores[comp_idx]
                if self.key not in comp_ignore:
                    comp_ignore = comp_ignore + [self.key]
                filtered_comp = []
                for f in comp:
                    if isinstance(f, str):
                        f_key = f
                        # Also check the resolved function's .key (may differ after merges)
                        if isinstance(lex, dict) and f in lex:
                            resolved_key = lex[f].key
                        else:
                            resolved_key = f
                    else:
                        f_key = to_key(f, kenntnis_lexikon=lex, fallback=False)
                        resolved_key = f_key
                    if f_key not in comp_ignore and resolved_key not in comp_ignore:
                        filtered_comp.append(f)
                ir_filtered.append(filtered_comp)
            all_output = []
            for inputvector in cartesian_product(ir_filtered):
                # Build per-component ignore lists for the inserted function.
                # After insert, new components come from inp_0's sub-comps, then inp_1's, etc.
                # Each sub-component of inp_i inherits comp_i's ignore + self.key + key(inp_i).
                scf_inputs = [to_scf(e, kenntnis_lexikon=lex) for e in inputvector]
                next_component_ignores = []
                for comp_idx, si in enumerate(scf_inputs):
                    si_key = to_key(si, kenntnis_lexikon=lex, fallback=False)
                    # Include both the dictionary key and the resolved .key to prevent cycles
                    orig = inputvector[comp_idx]
                    orig_key = orig if isinstance(orig, str) else to_key(orig, kenntnis_lexikon=lex, fallback=False)
                    base_ignore = component_ignores[comp_idx] + [self.key, si_key]
                    if orig_key not in base_ignore:
                        base_ignore = base_ignore + [orig_key]
                    for _ in range(si.dimension()):
                        next_component_ignores.append(base_ignore)
                try:
                    new_outputs = self.insert(scf_inputs).all_outputs(only_confirmed=only_confirmed, component_ignores=next_component_ignores, _depth=_depth+1)
                except RecursionError:
                    print(component_ignores)
                    print(self.key)
                    print([to_key(i, fallback=False) for i in inputvector])   
                    raise RecursionError("Recursion depth exceeded during all_outputs. This likely means there is a cycle in the function definitions. Current function: " + self.key + ". Current input vector: " + str([to_key(i, fallback=False) for i in inputvector]))
                all_output += new_outputs 
            return all_output

    def all_outputs_as_voc(self, only_confirmed = False):
        ao = self.all_outputs(only_confirmed)
        aov = []
        for scf in ao:
            aov += [Vocabulary(scf.mapping[-1],scf.root,key=scf.key)]
        return aov

    def all_output_numbers(self, only_confirmed=False, ignore_keys=None, _depth=0):
        '''Return set of all numeric output values without constructing SCFunction objects.
        Much faster than all_outputs() for large lexicons.
        Uses simplified global ignore-set for cycle prevention.'''
        if _depth > 30:
            return set()
        if self.dimension() == 0:
            return {round(self.mapping[-1])}
        ir = self.confirmed_inputrange if only_confirmed else self.inputrange
        if ignore_keys is None:
            ignore_keys = frozenset()
        lex = self.kenntnis_lexikon
        my_ignore = ignore_keys | {self.key}
        comp_values = []
        for comp in ir:
            vals = set()
            for f in comp:
                f_scf = to_scf(f, kenntnis_lexikon=lex)
                f_key = f_scf.key
                dict_key = f if isinstance(f, str) else f_key
                if f_key in my_ignore or dict_key in my_ignore:
                    continue
                sub_vals = f_scf.all_output_numbers(only_confirmed, my_ignore | {f_key, dict_key}, _depth + 1)
                vals |= sub_vals
            comp_values.append(vals)
        if any(len(v) == 0 for v in comp_values):
            return set()
        result = set()
        for combo in itertools.product(*comp_values):
            val = self.mapping[-1]
            for i in range(len(combo)):
                val += self.mapping[i] * combo[i]
            result.add(round(val))
        return result

    def vereinige(self,mergee, other_mergees = [], printb= True,trust_affinity=True):
        # self = G_0
        # mergee = F^*
        # other_mergees = [G_2,G_3,...]
        merge_lex = self.kenntnis_lexikon if isinstance(self.kenntnis_lexikon, dict) else (mergee.kenntnis_lexikon if hasattr(mergee, 'kenntnis_lexikon') and isinstance(mergee.kenntnis_lexikon, dict) else globals().get('kenntnis_lexikon', None))
    
        # Behandle Errore
        for me in [mergee]+other_mergees:
            if type(other_mergees) != list:
                raise TypeError("Function multi_mergable requires a non-empty list of mergees")
            if not type(mergee) is SCFunction:
                print('not scf')
                raise TypeError("Can only merge with other SCFunctions") 
            if any(len(mergee.inputrange[comp])>1 for comp in range(self.dimension())):
                print('Mergee is not singleton')
                mergee.present()
                raise BaseException('Merge of SCFunctions is yet only implemented for mergees directly produced by proto_parse')
            #if self.dimension() == 0:
                #print('merger is not generalizable')
                #raise BaseException('SCFunction ' + self.root + ' of dimension 0 cannot merge')

        if printb:   
            print('Vereinige die folgenden Funktionen:')
            self.present()
            mergee.present()
            for me in other_mergees:
                me.present()

        # [U_1,...,U_k] =
        mergee_inputvector = [to_scf(mergee.inputrange[i][0]) for i in range(mergee.dimension())] # U=(U_1,...,U_k)

        # 11 Prüfe ob u bereits im affinen Spann des Definitionsbereichs von self liegt
        affine_basis = self.input_numberbase() # affine Basis des Defbereichs von self
        #print('affine_basis:', affine_basis)
        #print(np.array(affine_basis + [[m.mapping[-1] for m in mergee_inputvector] + [1]], dtype=float))
        if np.linalg.matrix_rank(np.array(affine_basis + [[m.mapping[-1] for m in mergee_inputvector] + [1]], dtype=float), tol=1e-3) == len(affine_basis):
            #print('mergee lies in affine span of inputrange of merger')
            if trust_affinity:

                #12 - 13
                vereinigte_funktion = SCFunction(self.root,[self.inputrange[comp] + mergee.inputrange[comp] for comp in range(self.dimension())], self.mapping, c = [self.confirmed_inputrange[comp] + mergee.inputrange[comp] for comp in range(self.dimension())],key=self.key, kenntnis_lexikon=merge_lex)
                
                # 14
                return vereinigte_funktion
            # Fehlerbehandlung bei nicht-linearer Zahlwortmorphologie
            else:
                # check if self predicts correct value for mergee
                insert=self.insert([to_scf(mergee.inputrange[comp][0]) for comp in range(mergee.dimension())])
                if round(insert.mapping[-1]) == round(mergee.mapping[-1]):
                    vereinigte_funktion = SCFunction(self.root,[self.inputrange[comp] + mergee.inputrange[comp] for comp in range(self.dimension())], self.mapping, c = [self.confirmed_inputrange[comp] + mergee.inputrange[comp] for comp in range(self.dimension())],key=self.key, kenntnis_lexikon=merge_lex)
                    return vereinigte_funktion
                else:
                    if printb:
                        print('Merging these functions would cause wrong value calculations, so the latter function is returned as an atom')
                        self.present()
                        mergee.present()
                    return mergee.insert([to_scf(mergee.inputrange[comp][0]) for comp in range(mergee.dimension())])

        # [W_2,W_3,...] =
        other_mergee_inputvectors = [[to_scf(me.inputrange[i][0]) for i in range(mergee.dimension())] for me in other_mergees] # W_2,W_3,...
        #print('other_mergee_inputvectors:', other_mergee_inputvectors)

        # 16 neue_dimension = l 
        neue_dimension = len([comp for comp in range(self.dimension()) if len(self.inputrange[comp]) > 1 or not set(mergee.inputrange[comp] + [omi[comp] for omi in other_mergee_inputvectors]).issubset(self.inputrange[comp])]) # l
        #print('neue_dimension',neue_dimension)
        #print('affine_basis',len(affine_basis))

        # 18 - 20
        # Konstruiere ersten Teil von \vec{y}
        bildwerte = [np.dot(self.mapping, basevec) for basevec in affine_basis]
        # Finde b_1,...,b_l
        other_benotigte_vektoren = neue_dimension - len(affine_basis)
        for other_indices in itertools.combinations(range(len(other_mergee_inputvectors)), other_benotigte_vektoren):
            other_vektoren = [other_mergee_inputvectors[i] for i in other_indices]
            neue_affine_basis = np.array(affine_basis + [[m.mapping[-1] for m in mergee_inputvector] + [1]] + [[m.mapping[-1] for m in other_inputvector] + [1] for other_inputvector in other_vektoren], dtype=float) # prüfe affin lineare unabhängigkeit
            rang_neue_affine_basis = np.linalg.matrix_rank(neue_affine_basis, tol=1e-7)
            # Konstruiere Rest von \vec{y} 
            bildwerte_erweitert = bildwerte + [mergee.mapping[-1]] + [other_mergees[i].mapping[-1] for i in other_indices]
            # Prüfe Bedingung für die b_1,...,b_l
            if rang_neue_affine_basis - 1 == neue_dimension:

                # 21
                koeffizienten = intlinsolve([b[:-1] for b in neue_affine_basis],bildwerte_erweitert)
                #print(koeffizienten)

                # 22 - 31
                neuer_definitionsbereich = [self.inputrange[comp] + mergee.inputrange[comp] + [me[comp] for me in other_vektoren] for comp in range(self.dimension())]
                neuer_bestatigter_definitionsbereich = [self.confirmed_inputrange[comp] + mergee.inputrange[comp] + [me[comp] for me in other_vektoren] for comp in range(self.dimension())]
                return SCFunction(self.root,neuer_definitionsbereich, list(koeffizienten), c = neuer_bestatigter_definitionsbereich, key=self.key, kenntnis_lexikon=merge_lex)
        
        # 33
        return mergee
         
    
    def present(self, domain=True, printout=True, only_confirmed = False):
        if self.dimension() == 0:
            if printout: 
                print(self.root+" = "+str(self.mapping[-1]))
            return self.root+" = "+str(self.mapping[-1])
        else:
            domainstrs = []
            if only_confirmed:
                ir = self.confirmed_inputrange
            else:
                ir = self.inputrange
            for comp in ir:
                component = '{'
                if len(comp) < 20:
                    for entry in comp:
                        if to_scf(entry).dimension() == 0:
                            component += str(to_scf(entry).mapping[-1])+','
                        else:
                            component += str(to_scf(entry).root)+','
                else:
                    for entry in comp[:10]:
                        if to_scf(entry).dimension() == 0:
                            component += str(to_scf(entry).mapping[-1])+','
                        else:
                            component += str(to_scf(entry).root)+','
                    component += '...,'
                    for entry in comp[-10:]:
                        if to_scf(entry).dimension() == 0:
                            component += str(to_scf(entry).mapping[-1])+','
                        else:
                            component += str(to_scf(entry).root)+','
                component = component[:-1]+'}'
                domainstrs += [component]
            domainstr = 'x'.join(domainstrs)
            if self.dimension() == 1:
                inpstr = 'x'
                outpstr = str(round(self.mapping[0],2)) + 'x+' + str(round(self.mapping[1],2))
            else:
                variables = ['x','y','z','a','b','c','d','e','f','g','h']
                inpstr = '('
                outpstr = ''
                for comp in range(self.dimension()):
                    inpstr += variables[comp]+','
                    outpstr += str(round(self.mapping[comp],2)) + variables[comp] + '+'
                inpstr = inpstr[:-1] + ')'
                if self.mapping[-1] != 0:
                    outpstr += str(round(self.mapping[-1],2))
                else:
                    outpstr = outpstr[:-1]
            if domain:
                retstr = "Function " + self.root + "\t maps " + domainstr + "\t by " + inpstr + '\t -> ' + outpstr
            else:
                retstr = self.root + "\t maps " + inpstr + '\t -> ' + outpstr
            if printout:
                print(retstr)
            return retstr
    
    def max_schatzung(self):
        return round(sum(max([m,0]) for m in self.mapping) * max([2] + self.mapping[:-1]))

    def min_schatzung(self):
        if all(m>=0 for m in self.mapping):
            return round(sum(self.mapping))
        else:
            return round(sum([max([m,0]) for m in self.mapping])/2)

    def verstarke(self,lexikon,orakel,avoid_any_clash=False,printb=True):
        kenntnis_lexikon = self.kenntnis_lexikon if isinstance(self.kenntnis_lexikon, dict) else lexikon
        if self.dimension() == 0:
            return self
        candidates_for_abstraction = []

        # 11
        upper_limit = round(max([sum([max([0,coeff - 1]) for coeff in self.mapping[:-1]]), self.mapping[-1] - 1]))
        # Only confirmed inputs may raise the limit. Unconfirmed composite inputs (e.g. _und_zig with
        # confirmed_maximum 88) must not push upper_limit beyond the function's own range and thereby
        # admit nonsensical atoms (e.g. siebzig=70 into _undsiebzig with mapping [1,70]).
        for comp in range(self.dimension()):
            for entry in self.confirmed_inputrange[comp]:
                if to_scf(entry, kenntnis_lexikon=kenntnis_lexikon).confirmed_maximum > upper_limit:
                    upper_limit = to_scf(entry, kenntnis_lexikon=kenntnis_lexikon).confirmed_maximum
        #if printb: print('upper_limit: ',upper_limit)
        
        # 12 verstarkter_definitionsbereich = \mathcal{G}
        for key in lexikon.keys():
            entry = lexikon[key]
            if entry.maximum <= upper_limit:
                candidates_for_abstraction += [key]
            elif entry.confirmed_maximum < upper_limit and (orakel.style != 'manual'):
                candidates_for_abstraction += [key]
        verstarkter_definitionsbereich = [self.inputrange[comp].copy() for comp in range(self.dimension())]
        
        # 13
        for comp in range(self.dimension()):

            # 14
            if len(set(self.inputrange[comp])) > 1 or (isinstance(self.inputrange[comp][0], str) and any(len(c) > 1 for c in to_scf(self.inputrange[comp][0], kenntnis_lexikon=kenntnis_lexikon).inputrange)):

                #15
                for candidate in candidates_for_abstraction:

                    if not candidate in self.inputrange[comp]:

                        # 16
                        eingabekombinationen = []

                        # 17 - 21
                        nur_bestatigt = orakel.style in ['arithmetic', 'dummy', 'hybrid3']
                        for c in range(self.dimension()):
                            if c == comp:
                                eingabekombinationen += [to_scf(candidate).sample_outputs(10, only_confirmed=nur_bestatigt)]
                            else:
                                kombinationskomponente = []
                                for e in self.inputrange[c]:
                                    kombinationskomponente += to_scf(e).sample_outputs(5, only_confirmed=nur_bestatigt)
                                kombinationskomponente = random.sample(kombinationskomponente,min(len(kombinationskomponente),10))
                                eingabekombinationen += [kombinationskomponente]

                        entwurfe = []
                        klarer_fall = False
                        for eingabe in cartesian_product(eingabekombinationen):
                            entw = self.insert(list(eingabe))
                            # Use grammar_parse/grammar_generate instead of iterating all_outputs of all lexicon entries
                            # only_confirmed=True: only use confirmed knowledge to avoid blocking reinforcement based on unconfirmed inputs
                            parsed_number = grammar_parse(entw.root, kenntnis_lexikon, only_confirmed=True)
                            if parsed_number != -1 and round(parsed_number) == round(entw.mapping[-1]):
                                klarer_fall = True
                                verstarkter_definitionsbereich[comp] += [candidate]
                                #if printb: print('Input '+to_scf(candidate).root+' added to input slot '+str(comp)+' of '+self.root)
                            elif parsed_number != -1 or grammar_generate(round(entw.mapping[-1]), kenntnis_lexikon, only_confirmed=True) != '':
                                klarer_fall = True
                            elif avoid_any_clash and grammar_generate(round(entw.mapping[-1]), kenntnis_lexikon, only_confirmed=False) != '':
                                klarer_fall = True
                            if klarer_fall:
                                break
                            entwurfe += [entw]
                        #print(entwurfe)

                        if not klarer_fall:

                            # 22
                            # Glatt: alle mapping-Koeffizienten außer comp (und Konstante) sind gerundet 0
                            glatt = all(round(self.mapping[c]) == 0 for c in range(len(self.mapping)) if c != comp)
                            if orakel.antwort(entwurfe, glatt=glatt, show_plot=False, template_scf=self, varied_component=comp):

                                # 23
                                verstarkter_definitionsbereich[comp] += [candidate]
                                #if printb: print('Input '+to_scf(candidate).root+' added to input slot '+str(comp)+' of '+self.root)
            
        verstarkte_funktion = SCFunction(self.root,verstarkter_definitionsbereich,self.mapping,self.confirmed_inputrange,key=self.key, kenntnis_lexikon=kenntnis_lexikon)
        
        # 28
        return verstarkte_funktion


class Vocabulary(SCFunction):
    def __init__(self, nb, nal, key=None):
        if type(nb) is list or type(nb) is np.array:
            try:
                nb=nb.item()
            except:
                pass
        if not type(nal) is str:
            raise TypeError("Numeral has to be a string")
        self.number = nb
        self.word = nal
        if key is None:
            self.key = nal
        else:
            self.key = key
        self.root = nal
        self.inputrange = []
        self.confirmed_inputrange = []
        self.mapping = [nb]
        # For compatibility with SCFunction logic
        self.minimum = nb
        self.maximum = nb
        self.confirmed_minimum = nb
        self.confirmed_maximum = nb
    def printVoc(self):
        print(str(self.number)+' '+self.word)
    def all_outputs(self, only_confirmed = False):
        return [self]
    def dimension(self):
        return 0
    def actual_dimension(self):
        return 1
    def sample(self):
        return self

class Oracle:
    def __init__(self, style, data=None, search_engine=[], estimation_parameters=[30,0], tolerance_factor = 10, tolerance_factor_glatt = None, tolerance_factor_nichtglatt = None, security_factor = 5, knowledge=None, negative_knowledge=None):
        if style not in ['statistic','manual','wortliste','allwissend','arithmetic', 'dummy','hybrid','hybrid2','hybrid3']:
            raise TypeError("Style must be either 'statistic', 'manual', 'arithmetic', 'dummy', 'hybrid', 'wortliste', or 'allwissend'.")
        if data is None:
            data = {}
        if knowledge is None:
            knowledge = []
        if negative_knowledge is None:
            negative_knowledge = []
        if style == 'wortliste' and knowledge == [] or style == 'wortliste' and type(knowledge) != list:
            raise ValueError('Wortliste supervisor requires a knowledge list.')
        if style == 'allwissend' and type(knowledge) != dict:
            raise ValueError('Allwissend supervisor requires a knowledge dict {zahl: wort}.')
        if style in ['statistic','hybrid'] and not isinstance(search_engine, types.FunctionType):
            raise TypeError('Statistic or hybrid supervisor require a method search_engine that maps input strings to a number of search results.')
        self.style = style
        self.data = data
        self.search_engine = search_engine
        self.estimation_parameters = estimation_parameters
        self.knowledge = knowledge
        self.negative_knowledge = negative_knowledge
        if self.style == 'statistic' and self.data != {}:
            self.update([])
        self.tolerance_factor = tolerance_factor
        # Für glatte/nichtglatte Entwürfe unterschiedliche Toleranzfaktoren (nur relevant bei style='statistic').
        # Wenn nicht explizit gesetzt, gilt jeweils tolerance_factor.
        self.tolerance_factor_glatt = tolerance_factor_glatt if tolerance_factor_glatt is not None else tolerance_factor
        self.tolerance_factor_nichtglatt = tolerance_factor_nichtglatt if tolerance_factor_nichtglatt is not None else tolerance_factor
        self.security_factor = security_factor
        self.fragen_total = 0
        self.fragen_jn = []
        self.fragen_offen = []
        self.antwort_log = []
        self.search_cache = {}
        self.log_dir = None  # Wenn gesetzt, werden Plots als Dateien in dieses Verzeichnis gespeichert
        self.html_log = None  # Wenn gesetzt (Objekt mit write_plot(fig)), werden Plots inline in HTML eingebettet
        
    def antwort(self,entwurfe, glatt = None, tol = 0,show_plot=True, template_scf=None, varied_component=None):
        # Optional logging context (only filled for the 'statistic' path):
        # - template_scf: the SCFunction whose insert(...) produced these Entwurfe (the "Schablone").
        # - varied_component: index of the input slot that was varied across the Entwurfe.
        # Both default to None for backward compatibility with all other call sites.
        if self.style in ['statistic','hybrid2']:
            for entwurf in entwurfe:
                if isinstance(entwurf,str):
                    raise TypeError("Statistic supervisor only answers to inputs SCFunction, Vocabulary, or [word, number]")
            l = len(entwurfe)
            if l > 20:
                l = 20
                entwurfe = random.sample(entwurfe,20)
            value = sum([p.mapping[-1] for p in entwurfe]) / l
            sr_list = []
            for p in entwurfe:
                if p.root not in self.search_cache:
                    self.search_cache[p.root] = self.search_engine(p.root)
                sr_list.append(self.search_cache[p.root])
            search_results = sum(sr_list) / l
            if self.style == 'hybrid2':
                search_results += 1  # für kleine sprachen mit wenig Daten etwas mehr Toleranz gewähren
            if tol == 0:
                if glatt is True:
                    tol = self.tolerance_factor_glatt
                elif glatt is False:
                    tol = self.tolerance_factor_nichtglatt
                else:
                    tol = self.tolerance_factor
            a, b = self.estimation_parameters
            expected = np.exp(a) * value ** b
            lower_limit = expected / tol
            ratio = search_results / expected if expected > 0 else 0
            log_entry = {'entwurfe': [p.root for p in entwurfe], 'value': value, 'search_results': search_results, 'expected': expected, 'ratio': ratio, 'tol': tol, 'accepted': search_results > lower_limit}
            # --- Schablonen-Metadaten fuer die Rekonstruktion von Entwurfen mitloggen ---
            if template_scf is not None:
                log_entry['template_root'] = template_scf.root
                log_entry['template_mapping'] = list(template_scf.mapping)
            if varied_component is not None:
                log_entry['varied_component'] = varied_component
            self.antwort_log.append(log_entry)

            # --- Ausführlicher Bericht ---
            print("=" * 70)
            print("ORACLE.ANTWORT — Ausführlicher Bericht")
            print("=" * 70)
            print(f"Entwürfe ({l} Stück): {[p.root for p in entwurfe]}")
            print(f"Mittlerer Zahlwert (value):  {value:.4f}")
            print(f"Suchergebnisse pro Entwurf:  {sr_list}")
            print(f"Mittlere Suchergebnisse:     {search_results:.2f}")
            print(f"--- Berechnung von expected ---")
            print(f"  Regressionsparameter: a = {a:.6f}, b = {b:.6f}")
            print(f"  expected = exp(a) * value^b = exp({a:.6f}) * {value:.4f}^{b:.6f}")
            print(f"           = {np.exp(a):.6f} * {value**b:.6f}")
            print(f"           = {expected:.6f}")
            print(f"  Toleranzfaktor: {tol}")
            print(f"  lower_limit = expected / tol = {expected:.2f} / {tol} = {lower_limit:.2f}")
            print(f"  ratio = search_results / expected = {search_results:.2f} / {expected:.2f} = {ratio:.6f}")
            print(f"Anzahl Datenpunkte für Regression: {len(self.data)}")
            accepted = search_results > lower_limit
            print(f"Entscheidung: {search_results:.2f} {'>' if accepted else '<'} {lower_limit:.2f} → {'AKZEPTIERT' if accepted else 'ABGELEHNT'}")
            print("=" * 70)

            # --- Doppelt-logarithmischer Plot ---
            fig, ax = plt.subplots(figsize=(10, 7))

            # Kurvenverlauf
            if len(self.data) > 0:
                x_min = max(min(list(self.data.keys()) + [value]), 0.5)
                x_max = max(list(self.data.keys()) + [value]) * 2
            else:
                x_min = max(value * 0.1, 0.5)
                x_max = value * 10
            x_curve = np.logspace(np.log10(x_min), np.log10(x_max), 200)
            y_curve = np.exp(a) * x_curve ** b
            y_lower = y_curve / tol
            ax.plot(x_curve, y_curve, color='blue', linewidth=2, label=f'expected = $\\exp({a:.2f}) \\cdot x^{{{b:.2f}}}$')
            ax.plot(x_curve, y_lower, color='blue', linewidth=1, linestyle='--', label=f'lower\\_limit = expected / {tol}')
            ax.fill_between(x_curve, y_lower, y_curve, alpha=0.08, color='blue')

            # Datenpunkte der Regression (aus self.data)
            if len(self.data) > 0:
                data_x = list(self.data.keys())
                data_y = list(self.data.values())
                ax.scatter(data_x, data_y, color='grey', s=40, zorder=3, label=f'Regressionsdaten ({len(data_x)} Punkte)')

            # Frühere Entwurf-Datenpunkte aus antwort_log (blass, beeinflussen Kurve nicht)
            prev_entries = self.antwort_log[:-1]  # alle außer dem aktuellen
            if prev_entries:
                prev_values = [e['value'] for e in prev_entries]
                prev_sr = [e['search_results'] for e in prev_entries]
                prev_acc = [e['accepted'] for e in prev_entries]
                # akzeptierte vs abgelehnte in verschiedenen blassen Farben
                acc_v = [v for v, a in zip(prev_values, prev_acc) if a]
                acc_s = [s for s, a in zip(prev_sr, prev_acc) if a]
                rej_v = [v for v, a in zip(prev_values, prev_acc) if not a]
                rej_s = [s for s, a in zip(prev_sr, prev_acc) if not a]
                if acc_v:
                    ax.scatter(acc_v, acc_s, color='green', alpha=0.25, s=25, zorder=2, marker='o', label=f'Fr\"uhere Entw\"urfe akz. ({len(acc_v)})')
                if rej_v:
                    ax.scatter(rej_v, rej_s, color='red', alpha=0.25, s=25, zorder=2, marker='x', label=f'Fr\"uhere Entw\"urfe abg. ({len(rej_v)})')

            # Aktueller Entwurf-Datenpunkt
            marker_color = 'green' if accepted else 'red'
            ax.scatter([value], [search_results], color=marker_color, s=120, zorder=5, edgecolors='black', linewidths=1.5,
                       label=f'Aktuell: value={value:.1f}, SR={search_results:.0f}')

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Zahlwert (value)", fontsize=12)
            ax.set_ylabel("Suchergebnisse", fontsize=12)
            ax.set_title(f"Oracle.antwort --- expected={expected:.2f}, SR={search_results:.0f}, ratio={ratio:.4f}", fontsize=13)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, which='both', alpha=0.3)
            plt.tight_layout()
            if self.html_log and hasattr(self.html_log, 'write_plot'):
                self.html_log.write_plot(fig)
                plt.close(fig)
            elif self.log_dir:
                plot_nr = len(self.antwort_log)
                plot_path = os.path.join(self.log_dir, f'antwort_{plot_nr:04d}.png')
                fig.savefig(plot_path, dpi=120)
                print(f"[Plot gespeichert: {plot_path}]")
                plt.close(fig)
            else:
                plt.show()

            if search_results == 0:
                return False
            return accepted
        
        elif self.style == 'manual':
            # Wähle einen zufälligen entwurf aus der liste aus
            entwurf = random.choice(entwurfe)
            if isinstance(entwurf, SCFunction) or isinstance(entwurf, Vocabulary):
                entwurf = [entwurf.root,entwurf.mapping[-1]]
            if isinstance(entwurf, list):
                entwurf = entwurf
            while True:
                antwort = input("Lautet das Zahlwort für " + entwurf[0] + " " + str(round(entwurf[1])) + "? (j/n): ").strip().lower()
                if antwort == "j":
                    return True
                elif antwort == "n":
                    return False
                else:
                    print("Bitte antworten Sie mit 'j' oder 'n'.")
            
        elif self.style == 'wortliste':
            # wähle einen zufälligen entwurf aus der liste aus
            entwurf = random.choice(entwurfe)
            if isinstance(entwurf, SCFunction) or isinstance(entwurf, Vocabulary):
                entwurf = entwurf.sample().root
            if isinstance(entwurf, list):
                entwurf = entwurf[0]
            return entwurf in self.knowledge
        elif self.style == 'allwissend':
            entwurf = random.choice(entwurfe)
            if isinstance(entwurf, SCFunction) or isinstance(entwurf, Vocabulary):
                wort = entwurf.sample().root
                zahl = round(entwurf.mapping[-1])
            elif isinstance(entwurf, list):
                wort = entwurf[0]
                zahl = round(entwurf[1])
            else:
                return False
            self.fragen_total += 1
            self.fragen_jn += [wort]
            return self.knowledge.get(zahl) == wort
        elif self.style == 'arithmetic' or self.style == 'dummy' or self.style == 'hybrid3':
            #Since the code only make s arithmetically plausible proposals, everything is excepted
            return True

        elif self.style == 'hybrid':
            copy = Oracle('statistic',search_engine=self.search_engine, estimation_parameters=self.estimation_parameters,tolerance_factor = self.tolerance_factor, security_factor = self.security_factor)
            if copy.antwort(entwurfe,tol = copy.tolerance_factor / copy.security_factor):
                return True
            elif not copy.answer(entwurf,tol = copy.tolerance_factor * copy.security_factor):
                return False
            else:
                copy = Oracle('manual')
                return copy.answer(proposal)
            
        elif self.style == 'arithmetic' or self.style == 'dummy' or self.style == 'hybrid3':
            #Since the code only make s arithmetically plausible proposals, everything is excepted
            return True
            
    def answer(self,proposal, tol = 0,show_plot=True):
        if self.style == 'statistic':
            if isinstance(proposal,str):
                raise TypeError("Statistic supervisor only answers to inputs SCFunction, Vocabulary, or [word, number]")
            if isinstance(proposal,Vocabulary):
                proposal = [proposal.root,proposal.mapping[-1]]
            if isinstance(proposal,list):
                value = proposal[1]
                if proposal[0] not in self.search_cache:
                    self.search_cache[proposal[0]] = self.search_engine(proposal[0])
                search_results = self.search_cache[proposal[0]]
                all_names = [proposal[0]]
            elif isinstance(proposal,SCFunction):
                proposals = proposal.all_outputs()
                l = len(proposals)
                if l > 20:
                    l = 20
                    proposals = random.sample(proposals,20)
                sr_list = []
                for p in proposals:
                    if p.root not in self.search_cache:
                        self.search_cache[p.root] = self.search_engine(p.root)
                    sr_list.append(self.search_cache[p.root])
                value = sum([p.mapping[-1] for p in proposals]) / l
                search_results = sum(sr_list) / l
                all_names = [p.root for p in proposals]
            if tol == 0:
                tol = self.tolerance_factor
            expected = np.exp(self.estimation_parameters[0]) * value ** self.estimation_parameters[1]
            lower_limit = expected / tol
            ratio = search_results / expected if expected > 0 else 0
            self.antwort_log.append({'entwurfe': all_names, 'value': value, 'search_results': search_results, 'expected': expected, 'ratio': ratio, 'tol': tol, 'accepted': search_results > lower_limit})
            if search_results == 0:
                print("0 < " + str(round(lower_limit,1)))
                return False
            if show_plot:
                fig_a, ax_a = plt.subplots()
                x = np.arange(min(self.data.keys()),max(list(self.data.keys())+[value]))
                y = np.exp(self.estimation_parameters[0]) * x ** self.estimation_parameters[1] / tol
                ax_a.plot(x,y,color='black')
                ax_a.scatter(self.data.keys(),self.data.values(),color='grey')
                ax_a.scatter([value],[search_results],color='black',s=30)
                ax_a.set_xscale("log")
                ax_a.set_yscale("log")
                if self.html_log and hasattr(self.html_log, 'write_plot'):
                    self.html_log.write_plot(fig_a)
                    plt.close(fig_a)
                elif self.log_dir:
                    plot_nr = len(self.antwort_log)
                    plot_path = os.path.join(self.log_dir, f'answer_{plot_nr:04d}.png')
                    fig_a.savefig(plot_path, dpi=120)
                    plt.close(fig_a)
                else:
                    plt.show()
            if search_results > lower_limit:
                print(str(search_results) + " > " + str(lower_limit))
                return True
            else:
                print(str(search_results) + " < " + str(lower_limit))
                return False
        
        elif self.style == 'manual':
            if isinstance(proposal, SCFunction) or isinstance(proposal, Vocabulary):
                proposal = [proposal.root,proposal.mapping[-1]]
            if isinstance(proposal, list):
                proposal = proposal
            while True:
                antwort = input("Lautet das Zahlwort für " + proposal[0] + " " + str(proposal[1]) + "? (j/n): ").strip().lower()
                if antwort == "j":
                    return True
                elif antwort == "n":
                    return False
                else:
                    print("Bitte antworten Sie mit 'j' oder 'n'.")
            
        elif self.style == 'wortliste':
            if isinstance(proposal, SCFunction) or isinstance(proposal, Vocabulary):
                proposal = proposal.sample().root
            if isinstance(proposal, list):
                proposal = proposal[0]
            return proposal in self.knowledge
        elif self.style == 'allwissend':
            self.fragen_total += 1
            if isinstance(proposal, SCFunction) or isinstance(proposal, Vocabulary):
                wort = proposal.sample().root
                zahl = round(proposal.mapping[-1])
            elif isinstance(proposal, list):
                wort = proposal[0]
                zahl = round(proposal[1])
            else:
                return False
            self.fragen_jn += [wort]
            return self.knowledge.get(zahl) == wort
        elif self.style == 'arithmetic' or self.style == 'dummy' or self.style == 'hybrid3':
            #Since the code only make s arithmetically plausible proposals, everything is excepted
            return True

        elif self.style == 'hybrid':
            copy = Oracle('statistic',search_engine=self.search_engine, estimation_parameters=self.estimation_parameters,tolerance_factor = self.tolerance_factor, security_factor = self.security_factor)
            if copy.answer(proposal,tol = copy.tolerance_factor / copy.security_factor):
                return True
            elif not copy.answer(proposal,tol = copy.tolerance_factor * copy.security_factor):
                return False
            else:
                copy = Oracle('manual')
                return copy.answer(proposal)
            
    def update(self,new_confirmed_words):
        if type(new_confirmed_words) != list or not all([isinstance(w,SCFunction) or isinstance(w,Vocabulary) for w in new_confirmed_words]):
            raise TypeError('Update requires a list of SCFunctions, Vocabularys, or [word, number]-lists as input.')
        for w in new_confirmed_words:
            if isinstance(w,Vocabulary) or isinstance(w,SCFunction):
                w = [w.root,w.mapping[-1]]
            self.data[w[1]] = self.search_engine(w[0])
        data = [[i,self.data[i]] for i in self.data.keys()]
        self.estimation_parameters = bayesian_regression(data)
        
def bayesian_regression(data):
    # Bestimme Koeffizienten a, b sodass die ZWHäufigkeit Y durch den Zahlwert N mit a*N^b angenähert werden kann.
    # Dabei wird davon ausgegangen dass b ca -1.6503571428571429 mit Varianz 0.5683887566137565 ist
    exp_guess = -1.6503571428571429
    exp_uncert2 = 0.5683887566137565
    freq_unaccuracy2 = np.log(100)**2
    
    A = np.array([[len(data), sum([np.log(dat[0] + 0.01) for dat in data])],
                  [sum([np.log(dat[0] + 0.01) / freq_unaccuracy2 for dat in data]), sum([np.log(dat[0] + 0.01)**2 / freq_unaccuracy2 for dat in data]) + 1/exp_uncert2]])

    c = np.array([sum([np.log(dat[1] + 0.01) for dat in data]), sum([np.log(dat[0] + 0.01) * np.log(dat[1] + 0.01) / freq_unaccuracy2 for dat in data]) + exp_guess/exp_uncert2])
    a,b = np.linalg.solve(A,c)
    return [a,b]
            

def _to_native(val):
    """Convert sympy/numpy scalar to Python int or float to avoid slow sympy round()."""
    try:
        iv = int(val)
        if abs(iv - float(val)) < 1e-9:
            return iv
        return float(val)
    except (TypeError, ValueError, OverflowError):
        return float(val)

def intlinsolve(base,image):
    #print('base: ',base)
    #print('image: ',image)
    try:
        solution = list(solve(base,image)[0])
        #print('solution is ',solution)
        return [_to_native(s) for s in solution]+[0]
    except IndexError:
        #print('constant needed')
        erweiterte_basis = [list(b)+[1] for b in base]
        #print(erweiterte_basis)
        #print(image)
        try:
            return [_to_native(s) for s in solve(erweiterte_basis,image)[0]]
        except NotImplementedError:
            #print('unique solution')
            #return [round(i) for i in np.dot(np.linalg.pinv(np.array([b+[1] for b in base], dtype=np.float64),rcond=1e-15),image)]
            return [float(i) for i in np.dot(np.linalg.pinv(np.array(erweiterte_basis, dtype=np.float64)),image)]
        except IndexError:
            #print('unique solution')
            #return [round(i) for i in np.dot(np.linalg.pinv(np.array([b+[1] for b in base], dtype=np.float64),rcond=1e-15),image)]
            return [float(i) for i in np.dot(np.linalg.pinv(np.array(erweiterte_basis, dtype=np.float64)),image)]
    except NotImplementedError:
        #print('unique solution')
        return [round(float(i)) for i in np.dot(np.linalg.pinv(np.array(base, dtype=np.float64)),image)]+[0]

def cartesian_product(listlist):
    if len(listlist)==0:
        #print('ERROR: empty input in product')
        return [[]]
    for liste in listlist:
        if not isinstance(liste,list):
            liste=[liste]
    cp=[[x] for x in listlist[0]]
    for liste in listlist[1:]:
        cp=[a+[b] for a in cp for b in liste]
    return cp


def delatinized(string):
    #print(string)
    ad = AlphabetDetector()
    if not ad.is_latin(string):
        if not ad.is_cyrillic(string):
            if ad.is_cyrillic(string[0]) and not ad.is_cyrillic(string[-1]):
                #print('first part is cyrillic')
                for point in range(len(string)):
                    if not ad.is_cyrillic(string[:point]):
                        return string[:point-2]
            elif not ad.is_cyrillic(string[0]) and ad.is_cyrillic(string[-1]):
                #print('last part is cyrillic')
                for point in reversed(range(len(string))):
                    if not ad.is_cyrillic(string[point:]):
                        return string[point+2:]
            elif ad.is_latin(string[0]):
                #print('first part is latin')
                for point in range(len(string)+1):
                    if not ad.is_latin(string[:point]):
                        return string[point-1:] 
            elif ad.is_latin(string[-1]):
                #print('last part is latin')
                for point in reversed(range(len(string)+1)):
                    if not ad.is_latin(string[point:]):
                        return string[:point+1]
            else:
                return string
        else:
            return string
    else:
        return string
  
def create_lexicon(language,set_limit=10**9,lower_limit=0,exclude_ambiguous_numerals=True):
    if exclude_ambiguous_numerals:
        if language == 'Dogrib':
            set_limit = min([90, set_limit])
        elif language == 'Makhuwa':
            set_limit = min([600, set_limit])
        elif language == 'Purepecha':
            set_limit = min([130, set_limit])
        elif language == 'Susu':
            set_limit = min([200, set_limit])
        elif language == 'Tunica':
            set_limit = min([200, set_limit])
        elif language == 'Yao':
            set_limit = min([600, set_limit])
        elif language == 'Yupik':
            set_limit = min([500, set_limit])
        elif language == 'tr':
            set_limit = min([11000, set_limit])
        elif language == 'Haida':
            # not ambiguous but wrong (80-89 hold wrong words)
            set_limit = min([80, set_limit])
    LEX=[]
    try:
        num2words(1, lang=language)
        for integer in range(lower_limit, set_limit): #list(range(1,1001))+[1002,1006,1100,1200,1206,7000,7002,7006,7100,7200,7206,10000,17000,17200,17206,20000,27000,27006,27200,27206]:
            if integer < set_limit and lower_limit < integer:
                try:
                    numeral=num2words(integer, lang=language)
                    voc=Vocabulary(integer,numeral)
                    LEX=LEX+[voc]
                except:
                    pass
        return LEX
    except:
        try:
            #lanu=pd.read_csv(r'C:\Users\ikm\OneDrive\Desktop\NumeralParsingPerformance\Languages&NumbersData\Numeral.csv', encoding = "utf_16", sep = '\t')
            lanu=pd.read_csv(r'Numeral.csv', encoding = "utf_16", sep = '\t')
            df=lanu[lanu['Language']==language]
            biscriptual=False
            if ' ' in df.iloc[0,2]:
                biscriptual=True
            for i in range(len(df)):
                if i+1 < set_limit:
                    numeral=df.iloc[i,2]
                    if numeral[0]==' ':
                        numeral=numeral[1:]
                    if numeral[-1]==' ':
                        numeral=numeral[:-1]
                    if language in ['Latin','Persian','Arabic']:
                        words=numeral.split(' ')
                        numeral=' '.join(iter(words[:-1]))
                    if language in ['Chuvash','Adyghe','Belarusian','Ukrainian','Ingush']:
                        words=numeral.split(' ')
                        numeral=' '.join(iter(words[:len(words)//2]))
                    # For biscriptual languages: extract native script (not Latin transliteration)
                    elif biscriptual and not language in ['Latin','Persian','Arabic','Chuvash','Adyghe']:
                        ad = AlphabetDetector()
                        # Check if Latin comes first (like Japanese) or second (like Armenian, Russian)
                        first_is_latin = False
                        for char in numeral:
                            if char != ' ':
                                first_is_latin = ad.is_latin(char)
                                break
                        
                        if first_is_latin:
                            # Latin first, native script second (e.g., Japanese "ichi 一")
                            # Find first non-Latin, non-space character and take from there
                            for point in range(len(numeral)):
                                if numeral[point] != ' ' and not ad.is_latin(numeral[point]):
                                    numeral = numeral[point:].strip()
                                    break
                        else:
                            # Native script first, Latin second (e.g., Armenian "մեկ mek")
                            # Find first Latin character and take everything before it
                            for point in range(len(numeral)+1):
                                char = numeral[point:point+1]
                                if char and ad.is_latin(char) and char != ' ':
                                    numeral = numeral[:point].strip()
                                    break
                        
                        # Fallback to old delatinized approach if extraction didn't work
                        if numeral == df.iloc[i,2].strip() or not numeral:
                            numeral=delatinized(df.iloc[i,2].strip())
                    #print(numeral)
                    numeral=numeral.replace('%',',')
                    #print(numeral)
                    voc=Vocabulary(i+1,numeral)
                    LEX=LEX+[voc]
            return LEX
        except:
            #pass
            if True:
                lanu=pd.read_csv(r'DVNum.csv', encoding = "utf_8", sep = ';')
                #print(lanu)
                #print(lanu.columns)
                df=lanu[lanu['Language']==language]
                for i in range(len(df)):
                    if i+1 < set_limit:
                        numeral=df.iloc[i,2]
                        if numeral[0]==' ':
                            numeral=numeral[1:]
                        if numeral[-1]==' ':
                            numeral=numeral[:-1]
                        voc=Vocabulary(i+1,unicodedata.normalize('NFC',numeral))
                        LEX=LEX+[voc]
                for voc in LEX:
                    pass
                    #voc.printVoc()
                    #print(len(voc.word))
                return LEX
            #except:
                #raise NotImplementedError("Language "+language+" is not supported or spelled differently")  

def proto_parse(number,numeral,kenntnis_lexikon,print_documentation,print_result): #parse a (int number,str numeral)-pair using (current) kenntnis_lexikon. boolean print_documentation toggles documentation printout. boolean print_result toggles result printout
    #print('parse '+numeral)
    if print_documentation: print('parse '+numeral+' '+str(number))
    lex1 = []
    if isinstance(kenntnis_lexikon, dict):
        for key in kenntnis_lexikon.keys():
            entry = kenntnis_lexikon[key]
            if isinstance(entry,Vocabulary):
                lex1 += [entry]
            else:
                lex1 += entry.all_outputs_as_voc()
    elif len(kenntnis_lexikon) != 0 and isinstance(kenntnis_lexikon[0],Vocabulary):
        lex1 = kenntnis_lexikon
    else:
        for entry in kenntnis_lexikon:
            if isinstance(entry,Vocabulary):
                lex1 += [entry]
            else: 
                lex1 += entry.all_outputs_as_voc()
    lex1=lex1+[Vocabulary(number,numeral)] # so the new word itself is found at the end
    checkpoint=0 # point from which parsing is finally performed already
    highlights=[] #list of highlights, initially empty
    for end in range(len(numeral)+1): #set end of the observed substring
        startrange=range(checkpoint,end) #start of observed string may lie between checkpoint and end
        for highlight in highlights:
            startrange=set(startrange)-set(highlight.hlrange()) # observed strings may not start inside present highlights. rather they have to fully contain a highlight or be disjoint with it
        startrange=sorted(list(startrange)) # so ints in startrange are sorted by size
        for start in startrange: # set start of observed substring of numeral
            subnum_found_at_this_end=False # boolean condition to break start-loop
            substring=numeral[start:end] #set observed substring
            if print_documentation: print('substring: ',substring)
            for entry in lex1: #browse current kenntnis_lexikon
                if entry.word==substring: #look if substring appears
                    subnum_found_at_this_end=True
                    if 2*entry.number<number: #highlighting condition
                        if print_documentation: print(substring+' <' + str(entry.number) + '/2')
                        #highlights=[highlight for highlight in highlights if not highlight.start>=start]# if a highlight is contained in new highlight, then remove it from list of highlights
                        for highlight in highlights[:]: #browse through present highlights
                            if highlight.start>=start: # if a highlight is contained in new highlight,...
                                if print_documentation: print("remove "+highlight.numeral)
                                highlights.remove(highlight) #then remove it from list of highlights
                        highlights=highlights+[Highlight(entry,start)] # add new highlight
                        if print_documentation: print('Unpacked: ['+','.join([str(highlight.numeral) for highlight in highlights])+']')
                    else:
                        if print_documentation: print(substring+' ≥' + str(entry.number) + '/2')
                        checkpoint=end
                        if print_documentation: print("Set checkpoint behind "+numeral[:checkpoint])
                    break # out of browsing the kenntnis_lexikon
            if subnum_found_at_this_end:
                break # out of start-loop
    root=numeral
    for highlight in reversed(highlights):
        root=root[0:highlight.start]+'_'+root[highlight.end():len(root)]
    decompstr=str(number)+'='+root+'('
    decompstr=decompstr+','.join([str(highlight.number) for highlight in highlights])
    #for highlight in highlights:
    #    decompstr=decompstr+str(highlight.number)+','
    decompstr=decompstr+')'
    if print_result: print(decompstr)
    return SCFunction(root,[[Vocabulary(highlight.number,highlight.numeral,key=highlight.key)] for highlight in highlights],[0 for highlight in highlights]+[number], kenntnis_lexikon=kenntnis_lexikon)

def advanced_parse(number, word, kenntnis_lexikon, print_doc, print_result):
    if print_doc: print('parse '+word+' '+str(number))
    lexicon1 = []
    if isinstance(kenntnis_lexikon, dict):
        for key in kenntnis_lexikon.keys():
            entry = kenntnis_lexikon[key]
            if isinstance(entry,Vocabulary):
                lexicon1 += [entry]
            else:
                lexicon1 += entry.all_outputs_as_voc()
    elif len(kenntnis_lexikon) != 0 and isinstance(kenntnis_lexikon[0],Vocabulary):
        lexicon1 = kenntnis_lexikon
    else:
        for entry in kenntnis_lexikon:
            if isinstance(entry,Vocabulary):
                lexicon1 += [entry]
            else: 
                lexicon1 += entry.all_outputs_as_voc()
    lexicon1 = lexicon1+[Vocabulary(number,word)]    
    checkpoint = 0
    highlights=[]
    mult_found=False
    for end in range(0, len(word)+1):
        startrange=set(range(checkpoint, end))
        for highlight in highlights:
            startrange=startrange-set(highlight.hlrange())
            #print('remove '+str(range(highlight[3]+1,len(highlight[0]))))
        startrange=sorted(list(startrange))
        #print('startrange='+str(startrange))
        for start in startrange:
            subnum_found_at_this_end=False
            substr=word[start:end]
            if print_doc: print('substring = '+str(substr))
            for entry in lexicon1:
                if substr == entry.word:
                    subnum_found_at_this_end=True
                    subentry_found = False
                    if 2*entry.number < number or mult_found:
                        if print_doc: print(substr+' is <' + str(number) + '/2')
                        for highlight in reversed(highlights):
                            if highlight.start >= start:
                                if print_doc: print("remove "+highlight.numeral)
                                highlights.remove(highlight)
                        highlights=highlights+[Highlight(entry,start)]
                        if print_doc: print('Unpacked: ',[highlight.numeral for highlight in highlights])
                    else: 
                        if print_doc: print(substr+" is ≥" + str(number) + '/2')
                        mult_found=True
                        checkpoint=end
                        potential_highlight = None
                        earliest_laterstart = start+1
                        for highlight in highlights:
                            if highlight.number**2 < entry.number:
                                earliest_laterstart = min(end,highlight.end()) #so factors remain untouched
                        potential_highlight = None
                        for laterstart in range(earliest_laterstart,end):
                            if print_doc: print('subnum = '+word[laterstart:end])
                            for subentry in lexicon1:
                                if word[laterstart:end] == subentry.word:
                                    if subentry.number**2 <= entry.number:
                                        if print_doc: 
                                            print(word[laterstart:end]+" is <sqrt("+str(entry.number)+")") #print(word[laterstart:end]+" is FAC or SUM. If it would contain mult, its square would be larger than "+entry.word+'.')
                                            if potential_highlight:
                                                print("Ignore " + potential_highlight.word + " because " + word[laterstart:end] + " is its subnumeral.")
                                        subentry_found=True
                                        #for highlight in reversed(highlights):
                                            #if highlight.end() > laterstart:
                                                #if print_doc: print("remove "+highlight[0])
                                                #highlights.remove(highlight)
                                        #highlights=highlights+[Highlight(Vocabulary(subentry.number,subentry.word),laterstart)]
                                        #checkpoint = laterstart
                                        potential_highlight = Highlight(subentry,laterstart)
                                        potential_checkpoint = laterstart
                                        if print_doc: print('Unpacked: ',[highlight.numeral for highlight in highlights])
                                    else:
                                        if entry.number % subentry.number != 0 and 2*subentry.number < number:
                                            if print_doc: 
                                                print(word[laterstart:end]+" is at least <" + str(number) + "/2 and is no divisor of " + entry.word + ".") #print(word[laterstart:end]+" probably contains SUM. As "+subentry.word+' is no divisor of '+entry.word+', '+entry.word+' has to contain SUM. '+subentry.word+' cannot contain FAC*MULT, as it is smaller than half of '+entry.word+': And it cannot be FAC, as its square is larger than '+entry.word+'. So it is composed of SUM and MULT. If it turns out to be irreducible with the present properties, we assume it is SUM')
                                                if potential_highlight:
                                                    print("Ignore " + potential_highlight.word + " because " + word[laterstart:end] + " is its subnumeral.")
                                            potential_highlight = Highlight(subentry,laterstart)
                                            potential_checkpoint = laterstart
                                        elif entry.number % subentry.number == 0:
                                            if print_doc: print(str(subentry.number) + " is divisor of " + str(entry.number) + " and is ≥sqrt(" + str(entry.number) + ").")
                                            potential_highlight = None
                                        elif 2*subentry.number >= number:
                                            if print_doc: print(str(subentry.number) +" is at least <" + str(number) + "/2")
                                            potential_highlight = None
                            if subentry_found:
                                break  
                        if potential_highlight != None:
                            for highlight in reversed(highlights):
                                if highlight.end() > potential_checkpoint:
                                    if print_doc: print("remove "+highlight.root)
                                    highlights.remove(highlight)
                            highlights = highlights+[potential_highlight]
                            checkpoint = potential_checkpoint
                            if print_doc: print('Unpacked: ',[highlight.root for highlight in highlights])
                        if print_doc: print("Set checkpoint behind "+word[:checkpoint])
                    break                    
            if subnum_found_at_this_end:
                break
    #print('Unpacked subnums: ',len(highlights))
    if len(highlights) == 2:
        if highlights[0].number + highlights[1].number == number:
            sohi = sorted(highlights, key=lambda highlight: highlight.number)
            suspected_mult = sohi[-1]
            if print_doc: print("remove "+suspected_mult.root+' because ' + sohi[0].root + ' + ' + sohi[1].root + " = " + word + " and "+ sohi[0].root + ' > ' + sohi[1].root + "so it is probably mult.")
            highlights.remove(suspected_mult)
        elif highlights[0].number * highlights[1].number == number:
            sohi = sorted(highlights, key=lambda highlight: highlight.number)
            suspected_mult = sohi[-1]
            if print_doc: print("remove "+suspected_mult.root+' because ' + sohi[0].root + ' * ' + sohi[1].root + " = " + word + " and "+ sohi[0].root + ' > ' + sohi[1].root + "so it is probably mult.")
            highlights.remove(suspected_mult)
    elif len(highlights) == 3:
        sohi = sorted(highlights, key=lambda highlight: highlight.number)
        suspected_mult = sohi[-1]
        #print('suspected mult: ', suspected_mult.root)
        if suspected_mult.number**2 > number:
            #other_numbers = [highlight.number for highlight in highlights if highlight != suspected_mult]
            other_numbers = [highlight for highlight in highlights if highlight != suspected_mult]
            if other_numbers[0].number * suspected_mult.number + other_numbers[1].number == number:
                if print_doc: print("remove "+suspected_mult.root+ " because " + other_numbers[0].root + " * " + suspected_mult.root + " + " + other_numbers[1].root + " = " + word + " and " + suspected_mult.root + " > " + other_numbers[0].root + " so it is probably mult.")
                highlights.remove(suspected_mult)
            elif other_numbers[1].number * suspected_mult.number + other_numbers[0].number == number:
                if print_doc: print("remove "+suspected_mult.root+ " because " + other_numbers[1].root + " * " + suspected_mult.root + " + " + other_numbers[0].root + " = " + word + " and " + suspected_mult.root + " > " + other_numbers[1].root + " so it is probably mult.")
                highlights.remove(suspected_mult)
    elif len(highlights) > 3:
        suspected_mult = max(highlights, key=lambda highlight: highlight.number)
        if suspected_mult.number**2 > number:
            other_numbers = [highlight.number for highlight in highlights if highlight != suspected_mult]
            for suspected_factor in other_numbers:
                suspected_summand = sum(other_numbers)-suspected_factor
                if suspected_factor * suspected_mult.number + suspected_summand == number:
                    if print_doc: print("remove "+suspected_mult.root+' since it is probably mult.')
                    highlights.remove(suspected_mult)
                    break
        
    #print(str(highlights)+wort+' '+str(zahl))
    root=word
    for highlight in reversed(highlights):
        root=root[0:highlight.start]+'_'+root[highlight.end():len(root)]
    decompstr = str(number)+'='+root+'('+','.join([str(highlight.number) for highlight in highlights])+')'
    if print_result: print(decompstr)
    return SCFunction(root,[[Vocabulary(highlight.number,highlight.numeral,key=highlight.key)] for highlight in highlights],[0 for highlight in highlights]+[number], kenntnis_lexikon=kenntnis_lexikon)


def _iter_entry_words(entry, only_confirmed=True):
    """Yield (word, value) pairs for an inputrange entry without building large lists."""
    scf_entry = to_scf(entry)
    if isinstance(scf_entry, Vocabulary):
        yield (scf_entry.word, scf_entry.number)
    elif scf_entry.dimension() == 0:
        yield (scf_entry.root, round(scf_entry.mapping[-1]))
    else:
        for ou in scf_entry.all_outputs(only_confirmed=only_confirmed):
            yield (ou.root, round(ou.mapping[-1]))

def _match_word_to_scf(word, pos, rootparts, part_idx, inputrange, mapping, only_confirmed=False):
    """
    Recursively match word[pos:] against the pattern rootparts[part_idx] + gap + rootparts[part_idx+1] + ...
    Returns the accumulated numeric value (sum of mapping[i]*input_i contributions + constant) if match found, else None.
    This avoids the cartesian product by matching one component at a time with early termination.
    """
    rp = rootparts[part_idx]
    rp_len = len(rp)
    if word[pos:pos+rp_len] != rp:
        return None
    if part_idx == len(rootparts) - 1:
        # Last rootpart — must match exactly the remaining word
        if pos + rp_len == len(word):
            return mapping[-1]
        return None
    gap_start = pos + rp_len
    comp_idx = part_idx
    for entry in inputrange[comp_idx]:
        for (ew, ev) in _iter_entry_words(entry, only_confirmed=only_confirmed):
            ew_len = len(ew)
            if word[gap_start:gap_start+ew_len] == ew:
                rest = _match_word_to_scf(word, gap_start + ew_len, rootparts, part_idx + 1, inputrange, mapping, only_confirmed=only_confirmed)
                if rest is not None:
                    return mapping[comp_idx] * ev + rest
    return None

def grammar_parse(word,kenntnis_lexikon,only_confirmed=False):
    """
    Parse a word using the kenntnis_lexikon. Returns the numeric value if the word is recognised, -1 otherwise.
    Optimized: uses recursive pattern matching instead of enumerating all outputs via cartesian product.
    If only_confirmed=True, only confirmed inputrange entries are used.
    """
    if isinstance(kenntnis_lexikon, dict):
        lex_entries = [kenntnis_lexikon[key] for key in kenntnis_lexikon.keys()]
    else:
        lex_entries = kenntnis_lexikon
    for lex_entry in lex_entries:
        rootparts = lex_entry.root.split('_')
        # Quick check: all non-empty rootparts must appear in word
        if not all(rp in word for rp in rootparts if rp):
            continue
        # Select inputrange based on only_confirmed flag
        ir = lex_entry.confirmed_inputrange if only_confirmed else lex_entry.inputrange
        # Filter inputrange: only entries whose rootparts are all substrings of word
        filtered_ir = []
        scf_suitable = True
        for comp in ir:
            filtered_comp = [f for f in comp if all(rp in word for rp in to_scf(f).root.split('_') if rp)]
            if not filtered_comp:
                scf_suitable = False
                break
            filtered_ir.append(filtered_comp)
        if not scf_suitable:
            continue
        result = _match_word_to_scf(word, 0, rootparts, 0, filtered_ir, lex_entry.mapping, only_confirmed=only_confirmed)
        if result is not None:
            return round(result)
    return -1

def _find_gen_combo(target, rootparts, inputrange, mapping, comp_idx, accumulated, word_so_far, suffix_min, suffix_max, only_confirmed=False):
    """
    Recursively search for an input combination that produces the target number.
    Uses suffix min/max bounds for aggressive pruning.
    """
    if comp_idx == len(inputrange):
        total = accumulated + mapping[-1]
        if round(total) == round(target):
            return word_so_far + rootparts[-1]
        return None
    for entry in inputrange[comp_idx]:
        for (ew, ev) in _iter_entry_words(entry, only_confirmed=only_confirmed):
            new_acc = accumulated + mapping[comp_idx] * ev
            # Pruning: check if remaining components can still reach target
            total_min = new_acc + suffix_min[comp_idx + 1]
            total_max = new_acc + suffix_max[comp_idx + 1]
            if target < total_min - 0.5 or target > total_max + 0.5:
                continue
            result = _find_gen_combo(
                target, rootparts, inputrange, mapping, comp_idx + 1,
                new_acc, word_so_far + rootparts[comp_idx] + ew,
                suffix_min, suffix_max, only_confirmed=only_confirmed
            )
            if result is not None:
                return result
    return None

def grammar_generate(number,kenntnis_lexikon,only_confirmed=False,_return_source=False):
    """
    Generate a word for the given number using the kenntnis_lexikon.
    Optimized: uses recursive search with suffix min/max pruning instead of enumerating all outputs.
    If only_confirmed=True, only confirmed inputrange entries are used.
    If _return_source=True, returns (word, root_key) instead of just word.
    """
    if isinstance(kenntnis_lexikon, dict):
        lex_entries = list(kenntnis_lexikon.items())
    else:
        lex_entries = [(e.root, e) for e in kenntnis_lexikon]
    lex_entries = sorted(lex_entries, key=lambda kv: kv[1].dimension())
    for lex_key, lex_entry in lex_entries:
        # Use confirmed bounds when only_confirmed is set
        if only_confirmed:
            if number < lex_entry.confirmed_minimum or number > lex_entry.confirmed_maximum:
                continue
        else:
            if number < lex_entry.minimum or number > lex_entry.maximum:
                continue
        if lex_entry.dimension() == 0:
            if round(lex_entry.mapping[-1]) == round(number):
                return (lex_entry.root, lex_key) if _return_source else lex_entry.root
            continue
        ir = lex_entry.confirmed_inputrange if only_confirmed else lex_entry.inputrange
        rootparts = lex_entry.root.split('_')
        # Precompute suffix min/max for pruning
        dim = len(ir)
        suffix_min = [0.0] * (dim + 1)
        suffix_max = [0.0] * (dim + 1)
        suffix_min[dim] = lex_entry.mapping[-1]
        suffix_max[dim] = lex_entry.mapping[-1]
        for j in range(dim - 1, -1, -1):
            comp_lo = min(to_scf(e).minimum for e in ir[j])
            comp_hi = max(to_scf(e).maximum for e in ir[j])
            if lex_entry.mapping[j] >= 0:
                suffix_min[j] = suffix_min[j+1] + lex_entry.mapping[j] * comp_lo
                suffix_max[j] = suffix_max[j+1] + lex_entry.mapping[j] * comp_hi
            else:
                suffix_min[j] = suffix_min[j+1] + lex_entry.mapping[j] * comp_hi
                suffix_max[j] = suffix_max[j+1] + lex_entry.mapping[j] * comp_lo
        result = _find_gen_combo(number, rootparts, ir, lex_entry.mapping, 0, 0.0, '', suffix_min, suffix_max, only_confirmed=only_confirmed)
        if result is not None:
            return (result, lex_key) if _return_source else result
    return ('', None) if _return_source else ''


def _fmt_coeff(c):
    cr = round(c)
    return str(int(cr)) if abs(c - cr) < 1e-6 else str(c)


def explain_generate(number, kenntnis_lexikon, _depth=0, _seen=None):
    """Wie grammar_generate, gibt aber zusaetzlich Baum- und Wertaufschluesselung zurueck.

    Returns (word, tree_repr, value_repr) oder None, wenn die Zahl nicht
    erzeugbar ist. tree_repr hat die Form 'root(child1,child2,...)' (bei dim>0)
    bzw. 'root' (bei dim=0). value_repr hat die Form 'const+(v1)*c1+(v2)*c2+...'.
    """
    if _depth > 30:
        return None
    if _seen is None:
        _seen = frozenset()
    if isinstance(kenntnis_lexikon, dict):
        lex_entries = list(kenntnis_lexikon.values())
    else:
        lex_entries = list(kenntnis_lexikon)
    for lex_entry in lex_entries:
        if number < lex_entry.minimum or number > lex_entry.maximum:
            continue
        if lex_entry.dimension() == 0:
            if round(lex_entry.mapping[-1]) == round(number):
                return (lex_entry.root, lex_entry.root, _fmt_coeff(lex_entry.mapping[-1]))
            continue
        if lex_entry.key in _seen:
            continue
        result = _explain_combo_search(number, lex_entry, kenntnis_lexikon, _depth, _seen | {lex_entry.key})
        if result is not None:
            return result
    return None


def _explain_combo_search(target, lex_entry, lex, depth, seen):
    ir = lex_entry.inputrange
    mapping = lex_entry.mapping
    rootparts = lex_entry.root.split('_')
    dim = len(ir)
    suffix_min = [0.0] * (dim + 1)
    suffix_max = [0.0] * (dim + 1)
    suffix_min[dim] = mapping[-1]
    suffix_max[dim] = mapping[-1]
    for j in range(dim - 1, -1, -1):
        comp_lo = min(to_scf(e).minimum for e in ir[j])
        comp_hi = max(to_scf(e).maximum for e in ir[j])
        if mapping[j] >= 0:
            suffix_min[j] = suffix_min[j+1] + mapping[j] * comp_lo
            suffix_max[j] = suffix_max[j+1] + mapping[j] * comp_hi
        else:
            suffix_min[j] = suffix_min[j+1] + mapping[j] * comp_hi
            suffix_max[j] = suffix_max[j+1] + mapping[j] * comp_lo
    return _explain_rec(target, lex_entry, ir, mapping, rootparts, 0, 0.0, '',
                        [], [], suffix_min, suffix_max, lex, depth, seen)


def _explain_rec(target, lex_entry, ir, mapping, rootparts, comp_idx, acc,
                 word_so_far, tree_children, val_children,
                 suffix_min, suffix_max, lex, depth, seen):
    if comp_idx == len(ir):
        if round(acc + mapping[-1]) == round(target):
            word = word_so_far + rootparts[-1]
            tree = lex_entry.root + '(' + ','.join(tree_children) + ')'
            parts = [_fmt_coeff(mapping[-1])]
            for i, vc in enumerate(val_children):
                parts.append('(' + vc + ')*' + _fmt_coeff(mapping[i]))
            return (word, tree, '+'.join(parts))
        return None
    for entry in ir[comp_idx]:
        for (ew, ev) in _iter_entry_words(entry, only_confirmed=False):
            new_acc = acc + mapping[comp_idx] * ev
            total_min = new_acc + suffix_min[comp_idx + 1]
            total_max = new_acc + suffix_max[comp_idx + 1]
            if target < total_min - 0.5 or target > total_max + 0.5:
                continue
            child = explain_generate(ev, lex, _depth=depth + 1, _seen=seen)
            if child is None or child[0] != ew:
                child_tree, child_val = ew, _fmt_coeff(ev)
            else:
                _, child_tree, child_val = child
            tree_children.append(child_tree)
            val_children.append(child_val)
            res = _explain_rec(target, lex_entry, ir, mapping, rootparts,
                               comp_idx + 1, new_acc,
                               word_so_far + rootparts[comp_idx] + ew,
                               tree_children, val_children,
                               suffix_min, suffix_max, lex, depth, seen)
            if res is not None:
                return res
            tree_children.pop()
            val_children.pop()
    return None

def _finde_eingaben_fuer_ausgabe(funktion, zielwert):
    """Findet eine Eingabekombination von funktion, die zielwert erzeugt.
    Gibt eine Liste von Einträgen (einer pro Komponente) zurück, oder None.
    Nutzt suffix_min/suffix_max-Pruning wie grammar_generate."""
    dim = funktion.dimension()
    if dim == 0:
        if round(funktion.mapping[-1]) == round(zielwert):
            return []
        return None
    ir = funktion.inputrange
    # Suffix-Schranken vorausberechnen (gleiche Logik wie grammar_generate)
    suffix_min = [0.0] * (dim + 1)
    suffix_max = [0.0] * (dim + 1)
    suffix_min[dim] = funktion.mapping[-1]
    suffix_max[dim] = funktion.mapping[-1]
    for j in range(dim - 1, -1, -1):
        comp_lo = min(to_scf(e).minimum for e in ir[j])
        comp_hi = max(to_scf(e).maximum for e in ir[j])
        if funktion.mapping[j] >= 0:
            suffix_min[j] = suffix_min[j+1] + funktion.mapping[j] * comp_lo
            suffix_max[j] = suffix_max[j+1] + funktion.mapping[j] * comp_hi
        else:
            suffix_min[j] = suffix_min[j+1] + funktion.mapping[j] * comp_hi
            suffix_max[j] = suffix_max[j+1] + funktion.mapping[j] * comp_lo
    return _finde_eingaben_rekursiv(funktion, zielwert, 0, 0.0, suffix_min, suffix_max)

def _finde_eingaben_rekursiv(funktion, zielwert, comp_idx, akkumulator, suffix_min, suffix_max):
    if comp_idx == funktion.dimension():
        if round(akkumulator + funktion.mapping[-1]) == round(zielwert):
            return []
        return None
    for eintrag in funktion.inputrange[comp_idx]:
        for (wort, wert) in _iter_entry_words(eintrag, only_confirmed=True):
            neuer_akku = akkumulator + funktion.mapping[comp_idx] * wert
            total_min = neuer_akku + suffix_min[comp_idx + 1]
            total_max = neuer_akku + suffix_max[comp_idx + 1]
            if zielwert < total_min - 0.5 or zielwert > total_max + 0.5:
                continue
            ergebnis = _finde_eingaben_rekursiv(funktion, zielwert, comp_idx + 1, neuer_akku, suffix_min, suffix_max)
            if ergebnis is not None:
                return [eintrag] + ergebnis
    return None

#%run numeral_decomposition_advanced.ipynb
def random_choice(my_dict):
    keys = list(my_dict.keys())
    weights = np.array(list(my_dict.values()), dtype=float)
    weights /= weights.sum()   # normalisieren zu Wahrscheinlichkeiten
    chosen = np.random.choice(keys, p=weights)
    return chosen

def shuffle_lexicon(lexicon):
    if not type(lexicon) == list:
        raise TypeError('Input lexicon must be of type list')
    #for entry in lexicon:
        #if not type(entry) in [__main__.Vocabulary, __main__.SCFunction]:
            #raise TypeError('All entries of the lexicon must be of type Vocabulary of SCFunction. The following entry ist not:' + str(entry))
    total_weight = sum([entry.number**(-2) if entry.number!=0 else 1 for entry in lexicon])
    numdict = {entry.number: entry.word for entry in lexicon}
    rounddict = {}
    for entry in lexicon:
        n = entry.number
        if n >= 1000:
            power = 10 ** int(np.log10(n))
            rounded = (n // power) * power
            rounddict[n] = rounded if rounded in numdict else n
        else:
            rounddict[n] = n
            length_of_rounded_word = len(entry.word)
            for nb in range(n // 2, 2 * n):
                if nb in numdict.keys() and numdict[nb] in entry.word and len(numdict[nb]) < length_of_rounded_word:
                    rounddict[n] = nb
                    length_of_rounded_word = len(numdict[nb])
    rounding_probability = 0.61
    freqdict = {
        entry.number: (1 - rounding_probability if entry.number == 0 else (1 - rounding_probability) * entry.number**(-2))
        for entry in lexicon
    }
    for nb in rounddict:
        if nb == 0:
            freqdict[rounddict[nb]] += rounding_probability
        else:
            freqdict[rounddict[nb]] += rounding_probability * nb**(-2)

    keys = list(freqdict.keys())
    weights = np.array(list(freqdict.values()), dtype=float)
    weights /= weights.sum()
    # Efraimidis-Spirakis weighted sampling without replacement: O(n log n)
    # Use log form for numerical stability: log(u^(1/w)) = log(u)/w
    u = np.random.uniform(size=len(keys))
    log_sort_keys = np.log(u + 1e-300) / weights
    order = np.argsort(-log_sort_keys)

    # Reuse existing Vocabulary objects instead of creating new ones
    entry_by_number = {entry.number: entry for entry in lexicon}
    shuffled_lexicon = [entry_by_number[int(keys[i])] for i in order]
    #for e in shuffled_lexicon:
        #e.printVoc()
    return shuffled_lexicon


def solve_lexicon_clashes(entry1, entry2, printb=True):
    if printb: print('Solving lexicon clashes between ' + entry1.root + ' and ' + entry2.root + '...')
    #if printb: entry1.present()
    #if printb: entry2.present()
    cio1 = [[[round(i) for i in inp], round(sum([inp[i]*entry1.mapping[i] for i in range(entry1.dimension())]) + entry1.mapping[-1])] for inp in cartesian_product(entry1.number_inputs(only_confirmed = True))]
    cio2 = [[[round(i) for i in inp], round(sum([inp[i]*entry2.mapping[i] for i in range(entry2.dimension())]) + entry2.mapping[-1])] for inp in cartesian_product(entry2.number_inputs(only_confirmed = True))]
    #print('cio1: ', cio1)
    #print('cio2: ', cio2)
    entwurfe1 = {}
    entwurfe1[entry1.key] = {}
    new_inputrange = []
    for comp in range(entry1.dimension()):
        entwurfe1[entry1.key][comp] = {}
        new_comp = []
        for entr in entry1.inputrange[comp]:
            if entr in entry1.confirmed_inputrange[comp] or len(entry1.inputrange[comp]) == 1:
                new_comp += [entr]
            else:
                eingabekombinationen = []
                for c in range(entry1.dimension()):
                    if c == comp:
                        eingabekombinationen += [[ou.mapping[-1] for ou in to_scf(entr).sample_outputs(only_confirmed=False)]]
                    else:
                        kombinationskomponente = []
                        for e in entry1.confirmed_inputrange[c]:
                            kombinationskomponente += [ou.mapping[-1] for ou in to_scf(e).sample_outputs(only_confirmed=True)]
                        eingabekombinationen += [kombinationskomponente]
                eingabe_io = []
                for eingabe in cartesian_product(eingabekombinationen):
                    eingabe_io += [[eingabe, round(sum([entry1.mapping[i]*eingabe[i] for i in range(entry1.dimension())]) + entry1.mapping[-1])]]
                if any(e[1] == i[1] for e in eingabe_io for i in cio2):
                    if printb: print(entry1.root + ' does not use the input ' + to_scf(entr).root + ' in component ' + str(comp) + ' because it clashes with confirmed value of ' + entry1.root)
                else:
                    new_comp += [entr]
                    entwurfe1[entry1.key][comp][to_key(entr)] = eingabe_io
        new_inputrange += [new_comp]
    entry1 = SCFunction(entry1.root, new_inputrange, entry1.mapping, c=entry1.confirmed_inputrange, key=entry1.key)

    entwurfe2 = {}
    entwurfe2[entry2.key] = {}
    new_inputrange = []
    for comp in range(entry2.dimension()):
        entwurfe2[entry2.key][comp] = {}
        new_comp = []
        for entr in entry2.inputrange[comp]:
            if entr in entry2.confirmed_inputrange[comp] or len(entry2.inputrange[comp]) == 1:
                new_comp += [entr]
            else:
                eingabekombinationen = []
                for c in range(entry2.dimension()):
                    if c == comp:
                        eingabekombinationen += [[ou.mapping[-1] for ou in to_scf(entr).sample_outputs(only_confirmed=False)]]
                    else:
                        kombinationskomponente = []
                        for e in entry2.confirmed_inputrange[c]:
                            kombinationskomponente += [ou.mapping[-1] for ou in to_scf(e).sample_outputs(only_confirmed=True)]
                        eingabekombinationen += [kombinationskomponente]
                eingabe_io = []
                for eingabe in cartesian_product(eingabekombinationen):
                    eingabe_io += [[eingabe, round(sum([entry2.mapping[i]*eingabe[i] for i in range(entry2.dimension())]) + entry2.mapping[-1])]]
                if any(e[1] == i[1] for e in eingabe_io for i in cio1):
                    if printb: print(entry2.root + ' does not use the input ' + to_scf(entr).root + ' in component ' + str(comp) + ' because it clashes with confirmed value of ' + entry1.root)
                else:
                    new_comp += [entr]
                    entwurfe2[entry2.key][comp][to_key(entr)] = eingabe_io
        new_inputrange += [new_comp]
    entry2 = SCFunction(entry2.root, new_inputrange, entry2.mapping, c=entry2.confirmed_inputrange, key=entry2.key)
    #print('After removing inputs that clash with confirmed values of the other entry:')
    #entry1.present()
    #entry2.present()

    io1 = [([round(i) for i in inp], round(sum([inp[i]*entry1.mapping[i] for i in range(entry1.dimension())]) + entry1.mapping[-1])) for inp in cartesian_product(entry1.number_inputs())]
    io2 = [([round(i) for i in inp], round(sum([inp[i]*entry2.mapping[i] for i in range(entry2.dimension())]) + entry2.mapping[-1])) for inp in cartesian_product(entry2.number_inputs())]
    
    # entry1
    new_inputrange = []
    for comp in range(entry1.dimension()):
        new_comp = []
        for entr in entry1.inputrange[comp]:
            if entr in entry1.confirmed_inputrange[comp] or len(entry1.inputrange[comp]) == 1:
                new_comp += [entr]
            else:
                eingabe_io = entwurfe1[entry1.key][comp][to_key(entr)]
                clash_wins = 0
                clash_losses = 0
                break_now = False
                for v in io2:
                    for e in eingabe_io:
                        if round(v[1]) == round(e[1]):
                            input1 = e[0]
                            input2 = v[0]
                            confirmed_nums1 = [set(round(x) for x in comp) for comp in entry1.number_inputs(only_confirmed = True)]
                            confirmed_nums2 = [set(round(x) for x in comp) for comp in entry2.number_inputs(only_confirmed = True)]
                            unconfirmed_components1 = [input1[i] for i in range(len(input1)) if round(input1[i]) not in confirmed_nums1[i]]
                            unconfirmed_components2 = [input2[i] for i in range(len(input2)) if round(input2[i]) not in confirmed_nums2[i]]
                            if sum(unconfirmed_components2) < sum(unconfirmed_components1):
                                clash_losses += 1
                            elif sum(unconfirmed_components2) > sum(unconfirmed_components1):
                                clash_wins += 1
                            elif sum(unconfirmed_components2) == sum(unconfirmed_components1):
                                pass
                        if abs(clash_wins - clash_losses) > 3:
                            break_now = True
                            break
                    if break_now:
                        break
                if clash_wins >= clash_losses:
                    #if printb: print(entry1.root + ' keeps the input ' + to_scf(entr).root + ' in component ' + str(comp) + ' despite clashes against entries of ' + entry2.root + '. (wins=' + str(clash_wins) + ' losses=' + str(clash_losses) + ')')
                    new_comp += [entr]
                else:
                    pass
                    #if printb: print(entry1.root + ' does not use the input ' + to_scf(entr).root + ' in component ' + str(comp) + ' because it loses clash against ' + entry2.root + '. (wins=' + str(clash_wins) + ' losses=' + str(clash_losses) + ')')
        new_inputrange += [new_comp]
    entry1 = SCFunction(entry1.root, new_inputrange, entry1.mapping, c=entry1.confirmed_inputrange, key=entry1.key)

    # entry2
    new_inputrange = []
    for comp in range(entry2.dimension()):
        new_comp = []
        for entr in entry2.inputrange[comp]:
            if entr in entry2.confirmed_inputrange[comp] or len(entry2.inputrange[comp]) == 1:
                new_comp += [entr]
            else:
                eingabe_io = entwurfe2[entry2.key][comp][to_key(entr)]
                clash_wins = 0
                clash_losses = 0
                break_now = False
                for v in io1:
                    for e in eingabe_io:
                        if round(v[1]) == round(e[1]):
                            input2 = e[0]
                            input1 = v[0]
                            confirmed_nums2 = [set(round(x) for x in comp) for comp in entry2.number_inputs(only_confirmed = True)]
                            confirmed_nums1 = [set(round(x) for x in comp) for comp in entry1.number_inputs(only_confirmed = True)]
                            unconfirmed_components2 = [input2[i] for i in range(len(input2)) if round(input2[i]) not in confirmed_nums2[i]]
                            unconfirmed_components1 = [input1[i] for i in range(len(input1)) if round(input1[i]) not in confirmed_nums1[i]]
                            if sum(unconfirmed_components1) < sum(unconfirmed_components2):
                                clash_losses += 1
                            elif sum(unconfirmed_components1) > sum(unconfirmed_components2):
                                clash_wins += 1
                            elif sum(unconfirmed_components1) == sum(unconfirmed_components2):
                                pass
                        if abs(clash_wins - clash_losses) > 3:
                            break_now = True
                            break
                    if break_now:
                        break
                if clash_wins >= clash_losses:
                    #if printb: print(entry2.root + ' keeps the input ' + to_scf(entr).root + ' in component ' + str(comp) + ' despite clashes against entries of ' + entry1.root + '.')
                    new_comp += [entr]
                else:
                    pass
                    #if printb: print(entry2.root + ' does not use the input ' + to_scf(entr).root + ' in component ' + str(comp) + ' because it loses clash against ' + entry1.root + '.')
        new_inputrange += [new_comp]
    entry2 = SCFunction(entry2.root, new_inputrange, entry2.mapping, c=entry2.confirmed_inputrange, key=entry2.key)
    return entry1, entry2


def learn_lexicon(wortliste,orakel,initial_kenntnis_lexikon=None,avoid_any_clash=False,version='a',printb=True,normalize=True,restricted_merge=True):
    #if printb: print('Learning '+language)
    #supervisor = wortliste

    # Alg. 2, Z. 3
    kenntnis_lexikon = initial_kenntnis_lexikon if initial_kenntnis_lexikon is not None else {}

    globals()['kenntnis_lexikon'] = kenntnis_lexikon
    if isinstance(kenntnis_lexikon, dict):
        for key in kenntnis_lexikon.keys():
            if isinstance(kenntnis_lexikon[key], SCFunction):
                kenntnis_lexikon[key].kenntnis_lexikon = kenntnis_lexikon

    # some tracking variables for efficiency or analysis
    samples = 0
    orakel_errors = []
    minimum = 1
    known_numbers = set()  # Cache: Zahlen, die das Lexikon bereits generieren kann
    # One-step UNDO: snapshot taken just before each accepted word is processed.
    _undo_kenntnis_lexikon = None  # None = no undo available yet
    _undo_minimum = 1

    # Pre-sort oracle keys for fast scanning (allwissend mode)
    if orakel.style == 'allwissend':
        oracle_keys_sorted = sorted(orakel.knowledge.keys())

    # Alg. 2, Z. 4
    for voc in wortliste:
        
        überschriebene_keys = []

        # Alg. 2, Z. 5
        if orakel.style in ['manual','hybrid']:
            voc = frage_nach_zahlwort(kenntnis_lexikon,minimum=minimum)
            if voc == None:
                break
            minimum = voc.number
        elif orakel.style == 'allwissend':
            # Scan only oracle keys (via bisect), skip known_numbers
            idx = bisect.bisect_left(oracle_keys_sorted, minimum)
            found = False
            while idx < len(oracle_keys_sorted):
                candidate = oracle_keys_sorted[idx]
                idx += 1
                if candidate in known_numbers:
                    continue
                word = grammar_generate(candidate, kenntnis_lexikon)
                if word != '':
                    known_numbers.add(candidate)
                    continue
                # Cannot generate → learn from oracle
                orakel.fragen_total += 1
                orakel.fragen_offen += [candidate]
                voc = Vocabulary(candidate, orakel.knowledge[candidate])
                minimum = candidate + 1
                found = True
                break
            if not found:
                break
        elif orakel.style in ['hybrid2', 'hybrid3']:
            while True:
                _input_fn = _hybrid2_input_func if _hybrid2_input_func is not None else input
                antwort = _input_fn("Vokabel (Format Zahl Wort, z.B. 1 eins): ")
                if antwort == 'STOP':
                    raise StopLearning()
                elif antwort == 'UNDO':
                    if _undo_kenntnis_lexikon is not None:
                        # Restore lexicon to state BEFORE the last processed word.
                        kenntnis_lexikon = dict(_undo_kenntnis_lexikon)
                        globals()['kenntnis_lexikon'] = kenntnis_lexikon
                        for _uk in kenntnis_lexikon:
                            if isinstance(kenntnis_lexikon[_uk], SCFunction):
                                kenntnis_lexikon[_uk].kenntnis_lexikon = kenntnis_lexikon
                        minimum = _undo_minimum
                        _undo_kenntnis_lexikon = None  # single-step undo: clear after use
                        print('[Undo: letztes Wort rückgängig gemacht]')
                    else:
                        print('[Kein Undo-Schritt verfügbar]')
                    continue
                elif ' ' in antwort:
                    teile = antwort.split(' ', 1)
                    try:
                        zahl = int(teile[0].strip())
                        wort = teile[1].strip().lower()
                        if wort:
                            # Save snapshot BEFORE this word is processed.
                            _undo_kenntnis_lexikon = dict(kenntnis_lexikon)
                            _undo_minimum = minimum
                            voc = Vocabulary(zahl, wort)
                            minimum = zahl + 1
                            break
                        else:
                            print('Bitte geben Sie ein Wort nach der Zahl ein.')
                    except ValueError:
                        print('Ungültige Eingabe. Format: Zahl Wort, z.B. 1 eins')
                else:
                    print('Ungültige Eingabe. Format: Zahl Wort, z.B. 1 eins')

        # Alg. 2, Z. 6 update statistical orakel based on data of new confirmed word
        if orakel.style in ['statistic','hybrid','hybrid2']:
            orakel.update([voc])

        #check if word is known already
        known = False
        if (type(orakel) == list or orakel.style not in ['dummy','arithmetic']):
            # Es könnte ein Problem werden, dass grammar_parse und grammar_generate nach dem ersten erfolg abbrechen und daher nicht alle möglichen bestätigten outputs der Einträge in learnerlex berücksichtigen. 
            if orakel.style in ['hybrid2', 'hybrid3']:
                # For hybrid2, only treat a word as known when there is already a confirmed path.
                # Abstract-only outputs do NOT count as known so the user can confirm them.
                if grammar_parse(voc.word, kenntnis_lexikon, only_confirmed=True) == voc.number or grammar_generate(voc.number, kenntnis_lexikon, only_confirmed=True) == voc.word:
                    known = True
            else:
                if grammar_parse(voc.word, kenntnis_lexikon) == voc.number or grammar_generate(voc.number, kenntnis_lexikon) == voc.word:
                    known = True
        if not known:

            #check if another word of voc's value has been learned before --> orakel error
            if orakel.style in ['statistic','hybrid'] and voc.mapping[-1] not in orakel.data.keys():
                for entr_key in kenntnis_lexikon.keys():
                    entr = kenntnis_lexikon[entr_key]
                    for o in entr.all_outputs():
                        if round(o.mapping[-1]) == round(voc.mapping[-1]):
                            print("ORAKEL ERROR: " + str(voc.mapping[-1]) + " apparently means " + voc.word + " and not " + o.word)
                            orakel_errors += [voc.mapping[-1]]
                        
            samples += 1
            #1 parse new word
            #if printb: print('What means '+str(voc.number)+'?')
            if printb: print('Orakel: '+voc.word+' means '+str(voc.number))

            # Alg. 2, Z. 7-8 parse = F^*
            if version == 'a':
                parse = advanced_parse(voc.number, voc.word, kenntnis_lexikon, False, printb) #[ou for e in kenntnis_lexikon for ou in kenntnis_lexikon[e].all_outputs()]
            
            elif version == 'p':
                parse = proto_parse(voc.number, voc.word, kenntnis_lexikon, False, printb) #[ou for e in kenntnis_lexikon for ou in kenntnis_lexikon[e].all_outputs()]

            # Understood?
            understood = False

            # Alg. 2, Z. 9
            functions_with_equal_template = [e for e in kenntnis_lexikon.values() if e.root == parse.root]
            if restricted_merge:
                mergable_functions = [e for e in functions_with_equal_template if parse.maximal_input < e.minimum and parse.minimum > e.maximal_input]
            else:
                mergable_functions = functions_with_equal_template
            
            # Alg. 2, Z. 10
            if mergable_functions != []:
                
                understood = True

                # Alg. 2, Z. 11-12
                merger = None
                for entry in mergable_functions:
                    if len(entry.number_inputs()) > 1:
                        merger = entry
                        mergable_functions.remove(entry)
                        break
                if merger == None:
                    merger = mergable_functions[0]
                    mergable_functions = mergable_functions[1:]

                # Alg. 2, Z. 13
                gelernte_funktion = merger.vereinige(parse, mergable_functions, printb=printb, trust_affinity=False) # versuche zu mergen
                
                # Alg. 2, Z. 15
                if merger.actual_dimension() < gelernte_funktion.actual_dimension(): # wenn mergen die dimension erweitert hat

                    # Alg. 2, Z. 16
                    gelernte_funktion = gelernte_funktion.verstarke(kenntnis_lexikon,orakel,avoid_any_clash=avoid_any_clash,printb=printb)

                # Alg. 2, Z. 19+21
                kenntnis_lexikon[gelernte_funktion.key] = gelernte_funktion

                # Alg. 2, Z. 18
                for key in kenntnis_lexikon.keys():
                    if key != gelernte_funktion.key:
                        sample = kenntnis_lexikon[key].sample_outputs(only_confirmed = True)
                        überdeckt = True
                        for s in sample:
                            if grammar_parse(s.root, [gelernte_funktion]) != round(float(s.mapping[-1])):
                                überdeckt = False
                                break
                        if überdeckt:
                            if printb: print('Overwrite ' + kenntnis_lexikon[key].root + ' because it is covered by ' + gelernte_funktion.root)
                            # Erweitere confirmed_inputrange der überdeckenden Funktion
                            if gelernte_funktion.dimension() > 0:
                                neue_confirmed = [list(gelernte_funktion.confirmed_inputrange[comp]) for comp in range(gelernte_funktion.dimension())]
                                for ausgabe in kenntnis_lexikon[key].all_outputs(only_confirmed=True):
                                    eingaben = _finde_eingaben_fuer_ausgabe(gelernte_funktion, round(float(ausgabe.mapping[-1])))
                                    if eingaben is not None:
                                        for comp in range(gelernte_funktion.dimension()):
                                            if eingaben[comp] not in neue_confirmed[comp]:
                                                neue_confirmed[comp].append(eingaben[comp])
                                gelernte_funktion = SCFunction(gelernte_funktion.root, gelernte_funktion.inputrange, gelernte_funktion.mapping, c=neue_confirmed, key=gelernte_funktion.key, kenntnis_lexikon=kenntnis_lexikon)
                                kenntnis_lexikon[gelernte_funktion.key] = gelernte_funktion
                            kenntnis_lexikon[key] = gelernte_funktion
                            überschriebene_keys += [key]
                
            # Alg. 2, Z. 23-25
            if not understood:
                #2b No, just remembering
                gelernte_funktion = parse
                kenntnis_lexikon[gelernte_funktion.key] = gelernte_funktion
            
            if printb: gelernte_funktion.present()

            updated_keys = []

            # Alg. 2, Z. 26-29 reinforce kenntnis_lexikon with gelernte_funktion
            for key in kenntnis_lexikon.keys():
                if kenntnis_lexikon[key].dimension() > 0: # sum(kenntnis_lexikon[key].mapping) > sum(gelernte_funktion.mapping) and
                    #if printb: print('Attempting to reinforce ' + kenntnis_lexikon[key].root + ' with ' + gelernte_funktion.root)
                    new_entry = kenntnis_lexikon[key].verstarke({gelernte_funktion.key: gelernte_funktion},orakel,avoid_any_clash=avoid_any_clash,printb=printb)
                    if any([len(new_entry.inputrange[j]) > len(kenntnis_lexikon[key].inputrange[j]) for j in range(new_entry.dimension())]):
                        kenntnis_lexikon[key] = new_entry
                        updated_keys += [new_entry.key]
                #else:
                    #new_entry = kenntnis_lexikon[i]
                #kenntnis_lexikon[i] = new_entry
                #if printb: entry.present()

            # Alg. 2, Z. 30-34 resolve lexicon clashes
            if orakel.style in ['arithmetic','hybrid2','hybrid3']:
                done_pairs = set()
                for key in kenntnis_lexikon.keys():
                    if key in überschriebene_keys:
                        continue
                    for ue_key in updated_keys + [gelernte_funktion.key]:
                        if key == ue_key:
                            continue
                        if kenntnis_lexikon[key] is kenntnis_lexikon[ue_key]:
                            continue
                        pair = frozenset({key, ue_key})
                        if pair in done_pairs:
                            continue
                        done_pairs.add(pair)
                        ue_min = kenntnis_lexikon[ue_key].minimum
                        ue_max = kenntnis_lexikon[ue_key].maximum
                        e_min = kenntnis_lexikon[key].minimum
                        e_max = kenntnis_lexikon[key].maximum
                        if ue_min <= e_max and ue_max >= e_min:
                            #print('Resolving clashes between ' + kenntnis_lexikon[key].root + ' and ' + kenntnis_lexikon[ue_key].root)
                            kenntnis_lexikon[key], kenntnis_lexikon[ue_key] = solve_lexicon_clashes(kenntnis_lexikon[key],kenntnis_lexikon[ue_key],printb=printb)
            
            # Alg. 2, Z. 28 überschreibe keys von überflüssigen einträgen im kenntnis_lexikon
            for ue in updated_keys:
                learned_outputs = kenntnis_lexikon[ue].all_outputs()
                for key in kenntnis_lexikon.keys():
                    if key == ue:
                        continue
                    # check if all confirmed outputs of entry are covered by learned 
                    all_covered = True
                    for ou in kenntnis_lexikon[key].all_outputs(only_confirmed = True):
                        covered = False
                        for lo in learned_outputs:
                            if [round(i) for i in ou.mapping] == [round(i) for i in lo.mapping] and ou.root == lo.root:
                                covered = True
                        if not covered:
                            all_covered = False
                            break
                    if all_covered:
                        #if printb:
                            #print('Überschreibe:')
                            #print(key,end=': ')
                            #kenntnis_lexikon[key].present()
                            #print('mit')
                            #print(ue,end=': ')
                            #kenntnis_lexikon[ue].present()
                        # Erweitere confirmed_inputrange der überdeckenden Funktion
                        überdeckende = kenntnis_lexikon[ue]
                        if überdeckende.dimension() > 0:
                            neue_confirmed = [list(überdeckende.confirmed_inputrange[comp]) for comp in range(überdeckende.dimension())]
                            for ausgabe in kenntnis_lexikon[key].all_outputs(only_confirmed=True):
                                eingaben = _finde_eingaben_fuer_ausgabe(überdeckende, round(float(ausgabe.mapping[-1])))
                                if eingaben is not None:
                                    for comp in range(überdeckende.dimension()):
                                        if eingaben[comp] not in neue_confirmed[comp]:
                                            neue_confirmed[comp].append(eingaben[comp])
                            kenntnis_lexikon[ue] = SCFunction(überdeckende.root, überdeckende.inputrange, überdeckende.mapping, c=neue_confirmed, key=überdeckende.key, kenntnis_lexikon=kenntnis_lexikon)
                        kenntnis_lexikon[key] = kenntnis_lexikon[ue]
                        überschriebene_keys.append(key)
            # 26, 31, 35 add learned to kenntnis_lexikon
            #kenntnis_lexikon += updated_entries                
            
            #print()
            #print('Entries: ')
            #for entry in kenntnis_lexikon.values():
                #if printb: entry.present()
            #print()
            
            #reorganize all inputranges so that they only contain scfunctions, no vocabulary
            if normalize:
                kenntnis_lexikon = normalize_scf_lexicon(kenntnis_lexikon,überschriebene_keys,printb)
                globals()['kenntnis_lexikon'] = kenntnis_lexikon
                for key in kenntnis_lexikon.keys():
                    if isinstance(kenntnis_lexikon[key], SCFunction):
                        kenntnis_lexikon[key].kenntnis_lexikon = kenntnis_lexikon

            # Rebuild known_numbers from lexicon after learning step
            # Only do full rebuild when structure changed (merge/reinforce)
            if orakel.style == 'allwissend':
                if understood or updated_keys:
                    known_numbers = set()
                    for _k in kenntnis_lexikon:
                        known_numbers |= kenntnis_lexikon[_k].all_output_numbers()
                else:
                    # Just memorized a new word — add its number directly
                    known_numbers.add(round(voc.mapping[-1]))

            #print(kenntnis_lexikon.keys())

    #if printb: print('Learned '+str(len(wortliste))+' words and structured them in '+str(len(kenntnis_lexikon))+' functions.')
    #if printb: print('It took '+str(samples)+' samples to learn those.')
    if printb and orakel.style == 'statistic': print(str(len(orakel_errors)) + ' orakel errors occurred. They affected the numbers ' + ', '.join(orakel_errors) + '.')
    if printb: print('Those are:')
    for key in kenntnis_lexikon.keys():
        entry = kenntnis_lexikon[key]
        if printb: 
            print(key + ': ',end='')
            entry.present()
            #print('Confirmed form: ',end='')
            #entry.present(only_confirmed=True)
    if printb: print('')
    return kenntnis_lexikon

def validate(kenntnis_lexikon, oracle):
    learned_words = []
    for key in kenntnis_lexikon.keys():
        e = kenntnis_lexikon[key]
        learned_words += [ou.root for ou in e.all_outputs()]
    mistake_found = False
    for ou in learned_words:
        if ou not in oracle:
            mistake_found = True
            print(ou, ' is wrong')
    for ou in oracle:
        if ou not in learned_words:
            mistake_found = True
            print(ou, ' is missing')
    if not mistake_found:
        print('No mistakes')


def normalize_scf_lexicon(kenntnis_lexikon,überschriebene_keys, printb= True):
    #for e in kenntnis_lexikon:
        #e.present()
    neues_kenntnis_lexikon = {}
    #print(überschriebene_keys)

    # Baue eine transitive Redirect-Map: A→B→C wird zu A→C
    redirect = {}
    for ük in überschriebene_keys:
        redirect[ük] = kenntnis_lexikon[ük].key
    changed = True
    while changed:
        changed = False
        for k in redirect:
            if redirect[k] in redirect:
                redirect[k] = redirect[redirect[k]]
                changed = True

    for key in kenntnis_lexikon.keys():
        eintrag = kenntnis_lexikon[key]
        #if printb: eintrag.present()
        new_inputrange = []
        new_c_inputrange = []
        for comp in range(eintrag.dimension()):
            if len(eintrag.inputrange[comp]) == 1 and not isinstance(eintrag.inputrange[comp][0],str):
                entry_obj = eintrag.inputrange[comp][0]
                if hasattr(entry_obj, 'key') and entry_obj.key in redirect:
                    entry_obj.key = redirect[entry_obj.key]
                new_comp = eintrag.inputrange[comp]
                c_entry_obj = eintrag.confirmed_inputrange[comp][0] if (len(eintrag.confirmed_inputrange[comp]) == 1 and not isinstance(eintrag.confirmed_inputrange[comp][0], str)) else None
                if c_entry_obj is not None and hasattr(c_entry_obj, 'key') and c_entry_obj.key in redirect:
                    c_entry_obj.key = redirect[c_entry_obj.key]
                new_c_comp = eintrag.confirmed_inputrange[comp]
            else:
                new_c_comp = []
                new_comp = []
                for ikey in eintrag.inputrange[comp]:
                    new_key = redirect.get(ikey, ikey)
                    if not new_key in new_comp:
                        new_comp += [new_key]
                for ikey in eintrag.confirmed_inputrange[comp]:
                    new_key = redirect.get(ikey, ikey)
                    if not new_key in new_c_comp:
                        new_c_comp += [new_key]
            new_inputrange += [new_comp]
            new_c_inputrange += [new_c_comp]
        neues_kenntnis_lexikon[key] = SCFunction(eintrag.root,new_inputrange,eintrag.mapping,new_c_inputrange,key=eintrag.key, kenntnis_lexikon=kenntnis_lexikon)

    # Entferne Phantom-Einträge: Dict-Key x zeigt auf SCFunction mit .key y (x≠y),
    # und y existiert auch als eigener Dict-Key → x ist nur ein verwaistes Duplikat.
    phantom_keys = [key for key in neues_kenntnis_lexikon
                    if neues_kenntnis_lexikon[key].key != key
                    and neues_kenntnis_lexikon[key].key in neues_kenntnis_lexikon]
    for key in phantom_keys:
        del neues_kenntnis_lexikon[key]

    # WICHTIG: Nach dem Rebuild zeigen alle Einträge auf das neue Lexikonobjekt.
    for key in neues_kenntnis_lexikon.keys():
        neues_kenntnis_lexikon[key].kenntnis_lexikon = neues_kenntnis_lexikon
    globals()['kenntnis_lexikon'] = neues_kenntnis_lexikon

    return neues_kenntnis_lexikon

def frage_nach_zahlwort(kenntnis_lexikon, minimum=1):
    while True:
        #print('Find word for '+str(minimum))
        word = grammar_generate(minimum,kenntnis_lexikon)
        #print(minimum,word)
        if word == '':
            while True:
                antwort = input("Wie lautet das Zahlwort für " + str(minimum) + "? ")
                if antwort == "":
                    print('Bitte geben Sie das Zahlwort für ' + str(minimum) + ' ein.')
                elif antwort == 'STOP':
                    raise StopLearning()
                elif antwort == 'UNDO':
                    raise UndoLearning()
                else:
                    return Vocabulary(minimum,antwort.strip().lower())

        minimum += 1
