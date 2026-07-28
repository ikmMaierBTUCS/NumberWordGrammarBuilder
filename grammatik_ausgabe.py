# -*- coding: utf-8 -*-
"""
Grammatik-Ausgabe: erzeugt aus einem gelernten kenntnis_lexikon eine CFG bzw.
eine MG und schreibt sie in eine Datei.

Herkunft
--------
scflist2cfg stammt aus These1Experiment/these5experiment.py, die komplette
CFG->MG-Maschinerie (cfg2mg & Co.) aus These1Experiment/mg_konversion.py.
Beide sind hier 1:1 uebernommen -- nur der Import zeigt auf das
funktionen_mit_referenzen DIESES Projekts, weil scflist2cfg mit isinstance auf
SCFunction prueft und to_scf sich ueber das modul-globale kenntnis_lexikon
abstimmt. Mit dem Modul des anderen Projekts waeren das verschiedene Klassen.

Die Lernphase aus numberset2MG faellt hier weg: die App hat ihr Lexikon schon,
wir brauchen nur die Phasen scflist2cfg und cfg2mg.

Zwei Ausgabeformate
-------------------
MG   -- unveraendert im Format von cfg2mg (Items mit Merkmalen + Semantik).
CFG  -- als NLTK-Feature-Grammatik (.fcfg). Siehe cfg_nach_nltk_fcfg.
"""

import io
import sys
import time
from itertools import combinations

from funktionen_mit_referenzen import SCFunction, to_scf


# ===========================================================================
# Teil 1: scflist2cfg  (1:1 aus these5experiment.py)
# ===========================================================================

def scflist2cfg(scflist, kenntnis_lexikon=None, diagnose=None, printb=False):
    # Eingabe pruefen
    for scf in scflist:
        if not isinstance(scf, SCFunction):
            raise BaseException('Grammar contains objects other than SCFunctions')

    def hole_scf(eintrag):
        return to_scf(eintrag, kenntnis_lexikon)

    # Kategorisiere die SCFunctions: jede Liste in 'categorization' ist eine
    # disjunkte Menge von SCFunctions, die eine Kategorie bilden.
    categorization = []
    copy = [scf for scf in scflist]

    if diagnose is not None:
        diagnose['phase'] = 'scflist2cfg'
        diagnose['in_while_schleife'] = True
        diagnose['while_iteration'] = 0
        diagnose['while_copy_len'] = len(copy)
        diagnose['while_ohne_fortschritt'] = False
        diagnose['while_letzter_fortschritt'] = time.time()

    while len(copy) > 0:
        next_highest_categ = []
        biggest_input = SCFunction('', [], [-10000])
        for scf in copy:
            for comp in scf.inputrange:
                for entr in comp:
                    entr = hole_scf(entr)
                    if scfbigger(entr, biggest_input) == 1:
                        biggest_input = entr
        if printb:
            print('Biggest input')
            biggest_input.present()
        for scfu in reversed(copy):
            if scfbigger(scfu, biggest_input) == 1:
                next_highest_categ += [scfu]
                copy.remove(scfu)
        categorization = [next_highest_categ] + categorization
        if printb:
            for s in next_highest_categ:
                s.present()

        if diagnose is not None:
            diagnose['while_iteration'] = diagnose['while_iteration'] + 1
            if len(copy) < diagnose['while_copy_len']:
                diagnose['while_letzter_fortschritt'] = time.time()
            diagnose['while_copy_len'] = len(copy)
            diagnose['while_ohne_fortschritt'] = (len(next_highest_categ) == 0)

        # Absicherung gegen die im Original dokumentierte Endlosschleife:
        # entfernt eine Iteration nichts, schrumpft copy nie mehr.
        if len(next_highest_categ) == 0:
            raise BaseException(
                'scflist2cfg kommt nicht voran (keine Funktion ist groesser als '
                'die groesste Eingabe) -- die Kategorisierung terminiert nicht.')

    if diagnose is not None:
        diagnose['in_while_schleife'] = False

    # Liste der Nichtterminale (NTs) bestimmen
    inputsets = []
    for scf in scflist:
        for comp in scf.inputrange:
            if not comp in inputsets:
                inputsets += [comp]
                if len(inputsets) > 24:
                    raise BaseException('Too many NTs required')

    # CFG-Regeln bauen
    CFG = []
    for scf in scflist:
        # Kategorie bestimmen
        for h in range(len(categorization)):
            if scf in categorization[h]:
                cat = 'c' + str(h + 1)
                break
        # enthaltene NTs bestimmen
        NTname = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                  'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        NTs = []
        for comp in scf.inputrange:
            for i in range(len(inputsets)):
                if comp == inputsets[i]:
                    NTs += [NTname[i]]
                    break
        # CFG-Wort zusammensetzen
        subwords = scf.root.split('_')
        CFGword = ''
        for j in range(len(NTs)):
            CFGword += subwords[j].lower() + NTs[j]
        CFGword += subwords[-1].lower()
        # Regeln machen
        CFG += [('S', CFGword, cat, scf.mapping)]
        for k in range(len(inputsets)):
            for entr in inputsets[k]:
                entr = hole_scf(entr)
                if entr.root == scf.root and entr.mapping == scf.mapping:
                    CFG += [(NTname[k], CFGword, cat, scf.mapping)]
                    break
    if printb:
        for r in CFG:
            print(r)
    return CFG


# ===========================================================================
# Teil 2: CFG->MG-Maschinerie  (1:1 aus mg_konversion.py)
# ===========================================================================

def scfbigger(x, y):
    '''
    return 1/0/-1 if x is bigger/equal/smaller than y
    '''
    xnumbers = [op.mapping[-1] for op in x.all_outputs()]
    ynumbers = [op.mapping[-1] for op in y.all_outputs()]
    if min(xnumbers) > max(ynumbers):
        return 1
    elif min(ynumbers) > max(xnumbers):
        return -1
    else:
        return 0


def find_best_antichain(items, timeout=-1):
    st = time.time()
    for length in reversed(range(len(items))):
        antichains_of_this_length = []
        for sublist in combinations(items, length):
            isAntiChain = True
            for comparison in combinations(sublist, 2):
                if comparison[0].issubset(comparison[1]) or comparison[1].issubset(comparison[0]):
                    isAntiChain = False
                    break
            if isAntiChain:
                antichains_of_this_length += [sublist]
            now = time.time()
            if timeout != -1 and now - st > timeout:
                return 'TimeOut'
        if antichains_of_this_length != []:
            antichain = antichains_of_this_length[0]
            for ac in antichains_of_this_length[1:]:
                if sum([len(i) for i in ac]) < sum([len(i) for i in antichain]):
                    antichain = ac
            return antichain


def feature_order(items, timeout=-1):
    items_with_schnitt = []
    for i in items:
        for j in items:
            schnitt = i.intersection(j)
            if not schnitt in items_with_schnitt:
                items_with_schnitt += [schnitt]

    print('items_with_schnitt: ', items_with_schnitt)

    antichain = find_best_antichain(items_with_schnitt, timeout=timeout)
    if antichain == 'TimeOut':
        return list(set([f for f in i for i in items]))
    print('antichain: ', antichain)

    subchains = {}
    superchains = {}
    for l in antichain:
        key = ''.join(list(l))
        subchains[key] = [l]
        superchains[key] = [l]
        for i in items_with_schnitt:
            if i.issubset(l):
                if i not in subchains[key]:
                    subchains[key] += [i]
            if i.issuperset(l):
                if i not in superchains[key]:
                    superchains[key] += [i]
        print(subchains[key])
        subchains[key].sort(key=len)
        superchains[key].sort(key=len)

    print('subchains :', subchains)
    print('superchains :', superchains)

    order = []

    for sbc in reversed(subchains.values()):
        for it in sbc:
            for f in it:
                if not f in order:
                    order += [f]

    for spc in reversed(superchains.values()):
        for it in spc:
            for f in it:
                if not f in order:
                    order += [f]

    print('order: ', order)

    return order


def cfg2mg(CFG, timeout=-1):
    '''
    Converts a categorized CFG into an MG
    Input: Categorized CFG = List of of rules
    Rule = 3-tuple (Non-Terminal, Word, Category),
    i.e. the Non-terminal can generate the word Word of category Category
    Capital letters are always interpretted as non-terminals here, minor letters as terminals.
    Words are strings out of terminals and non-terminals.
    Categories are strings that describe the type/category of a produced word.
    S is the default start symbol
    '''
    CFG += [('START', 'S', 'cFin', [1, 0])]
    N = set([rule[0] for rule in CFG])  # Set of all NTs
    F = []  # Free NTs
    R = []  # Restricted NTs
    target_category = {}  # Dictionary that stores the target category of each NT

    category_order = determine_partial_category_order(CFG)
    print(category_order.relations)
    aux_NTs = []
    for NT in N:
        # adapt the grammar so that NT has one unique target category
        # and write this in the dictionary 'target_category'
        NT_rules = [rule for rule in CFG if rule[0] == NT]
        targetted_categories = [rule[2] for rule in NT_rules]
        if len(targetted_categories) > 1:
            try:
                main_category = category_order.maximum(targetted_categories)
            except:
                print(NT)
                for rule in NT_rules:
                    print(rule)
                raise Exception('Could not unify target category')
            target_category[NT] = main_category
            updated_NT_rules = [rule for rule in NT_rules if rule[2] == main_category]
            secondary_categories = list(set(targetted_categories) - set([main_category]))
            for ind in range(len(secondary_categories)):
                category_order.relations.add((main_category, secondary_categories[ind]))
                updated_NT_rules += [(NT, NT + str(ind), main_category, [1, 0])]
                target_category[NT + str(ind)] = secondary_categories[ind]
                aux_NTs += [NT + str(ind)]
                for rule in NT_rules:
                    if rule[2] == secondary_categories[ind]:
                        updated_NT_rules += [(NT + str(ind), rule[1], secondary_categories[ind], rule[3])]
            CFG = [rule for rule in CFG if rule not in NT_rules] + updated_NT_rules
        else:
            target_category[NT] = targetted_categories[0]
        # decide if NT controls free input slots or restricted ones
        if free(NT, CFG):
            F += [NT]
        else:
            R += [NT]
    for n in aux_NTs:
        N.add(n)
        if free(n, CFG):
            F += [n]
        else:
            R += [n]

    transformable_CFG = []
    for rule in CFG:
        # decompose each rule into rules that are transformable into MG items
        # it means that they may only have NTs at their first or last spot and they have to be one of the shapes
        # word, Rword, wordF or FwordF
        transformable_CFG += decompose_rule(rule, R, F, N, target_category)

    summarized_tf_CFG = []
    for rule in transformable_CFG:
        same_rule_found = False
        for i in range(len(summarized_tf_CFG)):
            # check if 2 rules produce same string and category
            if rule[1].replace('*', '') == summarized_tf_CFG[i][1].replace('*', '') and rule[2] == summarized_tf_CFG[i][2]:
                same_rule_found = True
                # then: summarize them into one
                summarized_tf_CFG[i][0] += [rule[0]]
        if not same_rule_found:
            # otherwise: add new rule
            summarized_tf_CFG += [[[rule[0]], rule[1], rule[2], rule[3]]]

    MG_without_licensors = []
    for rule in summarized_tf_CFG:
        # create a proper MG item
        chars = list(rule[1])
        for char in reversed(range(len(chars))):
            if chars[char][0] == '*' or chars[char][0].isnumeric() or chars[char][0] == '+':
                chars = chars[:char - 1] + [chars[char - 1] + chars[char]] + chars[char + 1:]
        # produce semantic string
        # Die mappings tragen Float-Rauschen aus der linearen Algebra
        # (0.9999999999999998 statt 1). Fuer die Semantik-Strings wird auf ganze
        # Zahlen gerundet -- nur fuer die Ausgabe, rule[3] selbst bleibt
        # unangetastet, damit sich an der MG-Struktur nichts aendert.
        # Das repariert nebenbei die Fallunterscheidungen: '== [1, 0]' und
        # '== 0' treffen auf verrauschte Werte sonst nie zu.
        m = [int(round(k)) for k in rule[3]]
        if len(m) == 1:
            sem = ', ' + str(m[-1])
        elif len(m) == 2:
            if m == [1, 0]:
                sem = ', ' + "λ" + "x.x"
            elif m[-1] == 0:
                sem = ', ' + "λ" + "x.'" + str(m[0]) + "X'(x)"
            else:
                sem = ', ' + "λ" + "x.'" + str(m[0]) + "X+" + str(m[-1]) + "'(x)"
        elif len(m) == 3:
            if m[-1] == 0:
                sem = ', ' + "λ" + "x" + "λ" + "y.'" + str(m[1]) + "X+" + str(m[0]) + "Y'(x,y)"
            else:
                sem = ', ' + "λ" + "x" + "λ" + "y.'" + str(m[1]) + "X+" + str(m[0]) + "Y+" + str(m[-1]) + "'(x,y)"
        # compose full string
        if chars == []:
            item = ('[ :: ' + rule[2] + ']', rule[0], rule[2], rule[3], sem)
        elif chars[0] not in N and chars[-1] not in N:  # if word
            item = ('[' + ''.join(chars) + ' :: ' + rule[2] + ']', rule[0], rule[2], rule[3], sem)
        elif chars[0] in R:  # if Rword
            item = ('[' + ''.join(chars[1:]) + ' :: ' + '=' + target_category[chars[0]] + ', +' + chars[0] + ', ' + rule[2] + ']', rule[0], rule[2], rule[3], sem)
        elif (chars[0] not in N or len(chars) == 1) and chars[-1] in F:  # if wordF
            item = ('[' + ''.join(chars[:-1]) + ' :: ' + '=' + target_category[chars[-1]] + ', ' + rule[2] + ']', rule[0], rule[2], rule[3], sem)
        elif chars[0] in F and chars[-1] in F:  # if FwordF
            item = ('[' + ''.join(chars[1:-1]) + ' :: ' + '=' + target_category[chars[-1]] + ', ' + '=' + target_category[chars[0]] + ', ' + rule[2] + ']', rule[0], rule[2], rule[3], sem)
        MG_without_licensors += [item]

    # remove items whose category has no selector in the lexicon
    useless_items = []
    for item in MG_without_licensors:
        if item[2] == 'cFin':
            useless = False
        else:
            useless = True
            for other_item in MG_without_licensors:
                if '=' + item[2] in other_item[0] and other_item not in useless_items:
                    useless = False
                    break
        if useless:
            useless_items += [item]
    for item in useless_items:
        MG_without_licensors.remove(item)

    MG = []
    print([set([NT for NT in item[1] if NT in R]) for item in MG_without_licensors])
    fo = feature_order([set([NT for NT in item[1] if NT in R]) for item in MG_without_licensors])
    for item in MG_without_licensors:
        ordered_features = []
        for f in fo:
            if f in item[1]:
                ordered_features += [f]
        licensors = ['-' + NT for NT in ordered_features if NT in R]
        print('licensors of ' + item[0] + ':')
        print(licensors)
        new_item = (item[0][:-1] + ', ' + ', '.join(['' + licensor for licensor in licensors]) + ']' + item[4], licensors, item[2])
        MG += [new_item]

    all_licensors = set()
    all_licensor_chains = set()

    for item in MG:
        for licensor in range(len(item[1])):
            all_licensors.add((item[1][licensor], item[2]))
            if licensor < len(item[1]) - 1:
                all_licensor_chains.add((tuple(item[1][licensor:]), item[2]))

    for licensor in all_licensors:
        remove_shaper = ['[ :: =' + licensor[1] + ', +' + licensor[0][1:] + ', ' + licensor[1] + ']' + ', ' + "λ" + "x.x"]
        MG += [remove_shaper]

    for chain in all_licensor_chains:
        select_shaper = ['[ :: =' + chain[1] + ''.join([', +' + licensor[1:] for licensor in chain[0]]) + ', ' + chain[1] + ', ' + chain[0][0] + ']'', ' + "λ" + "x.x"]
        MG += [select_shaper]

    print('MG: ( ' + str(len(MG)) + ' items)')
    print('\n'.join([item[0] for item in MG]))
    return MG


def vari(string):
    return string + "*"


def contr(string):
    return string + '+'


def decompose_rule(rule, R, F, N, target_category):
    '''
    Input: one rule
    Output: list of rules that the rule is decomposed into
    '''
    symbol = rule[0]
    string = rule[1]
    mapping = rule[3]
    if type(string) == str:
        string = list(string)
        for char in reversed(range(len(string))):
            if string[char][0] == '*' or string[char][0].isnumeric():
                string = string[:char - 1] + [string[char - 1] + string[char]] + string[char + 1:]
    categ = rule[2]
    for spot in range(1, len(string) - 1):
        char = string[spot]
        if char in N:
            aux_NT = vari(char)
            aux_cat = vari(categ)
            while aux_NT in N:
                aux_NT = vari(aux_NT)
            while aux_cat in target_category.values():
                aux_cat = vari(aux_cat)
            N.add(aux_NT)
            F += [aux_NT]
            target_category[aux_NT] = aux_cat
            if string[0] in N:
                left_mapping = [mapping[0], 1, mapping[-1]]
                right_mapping = mapping[1:-1] + [0]
            else:
                left_mapping = [1, mapping[-1]]
                right_mapping = mapping[0:-1] + [0]
            return decompose_rule([symbol, string[:spot] + [aux_NT], categ, left_mapping], R, F, N, target_category) + decompose_rule([aux_NT, string[spot:], aux_cat, right_mapping], R, F, N, target_category)
    if len(string) > 1 and string[0] in R and string[-1] in R:
        F += [contr(string[0])]
        N.add(contr(string[0]))
        target_category[contr(string[0])] = categ + string[0]
        F += [contr(string[-1])]
        N.add(contr(string[-1]))
        target_category[contr(string[-1])] = categ + string[-1]
        return [[symbol, contr(string[0]) + ''.join(string[1:-1]) + contr(string[-1]), categ, mapping], [contr(string[0]), string[0], categ + string[0], [1, 0]], [contr(string[-1]), string[-1], categ + string[-1], [1, 0]]]
    elif len(string) > 1 and string[-1] in R:
        F += [contr(string[-1])]
        N.add(contr(string[-1]))
        target_category[contr(string[-1])] = categ + string[-1]
        return [[symbol, ''.join(string[:-1]) + contr(string[-1]), categ, mapping], [contr(string[-1]), string[-1], categ + string[-1], [1, 0]]]
    elif len(string) > 1 and string[0] in R and string[-1] in F:
        F += [contr(string[0])]
        N.add(contr(string[0]))
        target_category[contr(string[0])] = categ + string[0]
        return [[symbol, contr(string[0]) + ''.join(string[1:]), categ, mapping], [contr(string[0]), string[0], categ + string[0], [1, 0]]]
    elif len(string) > 1 and string[0] in F and string[-1] not in N:
        F += ['O']
        N.add('O')
        target_category['O'] = 'cnix'
        return [[symbol, ''.join(string) + 'O', categ, mapping[:-1] + [1] + [mapping[-1]]], ['O', '', 'cnix', [0]]]
    else:
        return [[rule[0], ''.join(rule[1]), rule[2], mapping]]


def free(NT, CFG):
    NT_rules = [NT_rule for NT_rule in CFG if NT_rule[0] == NT]
    for rule in CFG:
        # check that rule is not of NT but shares a common category with NT
        if rule not in NT_rules and rule[2] in [NT_rule[2] for NT_rule in NT_rules]:
            NT_rules_of_same_category = [NT_rule for NT_rule in NT_rules if NT_rule[2] == rule[2]]
            # check if in the category rule produces a word that NT does not
            if rule[1] not in [NT_rule[1] for NT_rule in NT_rules_of_same_category]:
                # then NT would be restricted
                return False
    return True


def restricted(NT, CFG):
    return not free(NT, CFG)


def determine_partial_category_order(CFG):
    N = set([rule[0] for rule in CFG])
    category_order = partial_order([])
    for rule in CFG:
        for char in rule[1]:
            if char in N:
                for rulec in CFG:
                    if rulec[0] == char:
                        category_order.relations.add((rule[2], rulec[2]))
    return category_order


class partial_order:
    def __init__(self, relations):
        self.relations = set(relations)

    def compare(self, a, b):
        if (a, b) in self.relations:
            return True
        for r in self.relations:
            if r[0] == a and self.compare(r[1], b):
                return True
        return False

    def add_rule(self, rule):
        self.relations.add(rule)

    def maximum(self, lst):
        max_element = None
        for element in lst:
            if not max_element or self.compare(element, max_element):
                max_element = element
        return max_element


# ===========================================================================
# Teil 3: CFG in NLTK-Syntax (.fcfg) -- mit semantischer Seite
# ===========================================================================
#
# Warum eine Feature-Grammatik und keine schlichte CFG?
# NLTK kennt zwei Ebenen:
#   nltk.CFG      -- rein syntaktisch, kein Platz fuer Semantik.
#   nltk.grammar.FeatureGrammar (.fcfg) -- Merkmalsstrukturen samt SEM-Merkmal,
#                    das Standardverfahren fuer Semantik im NLTK-Buch (Kap. 10).
# Nur die zweite Ebene kann den arithmetischen Teil ueberhaupt mittragen, darum
# schreiben wir .fcfg.
#
# Was NLTK dabei leistet und was nicht:
#   Die KOMPOSITION uebernimmt NLTK vollstaendig -- die SEM-Terme der Toechter
#   werden per Unifikation eingesetzt, Expression.simplify() beta-reduziert.
#   Das AUSRECHNEN nicht: nltk.sem.logic kennt keine Arithmetik, aus
#   plus(mal(10,4),3) wird kein 43. Der Term steht aber vollstaendig und korrekt
#   da; ihn auszuwerten ist ein Dreizeiler, den der Dateikopf mitliefert.
#
# Tokenisierung: die Regeln zerlegen ein Zahlwort in seine Morpheme. NLTK parst
# Token-Folgen, nicht Zeichenketten -- "dreiundvierzig" muss also als
# ['drei','und','vier','zig'] hereingereicht werden. Der Dateikopf zeigt, wie.

_NLTK_KOPF = """\
%% start S
#
# Zahlwort-Grammatik als NLTK-Feature-Grammatik (.fcfg)
# Automatisch erzeugt aus dem gelernten Kenntnis-Lexikon.
#
# Nichtterminale:  S = Startsymbol, N* = die gelernten Eingabebereiche.
# Terminale:       die Morpheme der Zahlwoerter.
# SEM:             der arithmetische Wert, kompositional aufgebaut aus
#                  plus(a,b) und mal(a,b).
#
# ---------------------------------------------------------------------------
# Verwendung
# ---------------------------------------------------------------------------
# NLTK parst Token-Folgen, keine Zeichenketten. Ein Zahlwort muss daher in
# seine Morpheme zerlegt werden -- die Terminale dieser Grammatik sind genau
# diese Morpheme.
#
#     import nltk
#     from nltk import FeatureChartParser
#     from nltk.grammar import FeatureGrammar
#
#     grammatik = FeatureGrammar.fromstring(open('%(dateiname)s', encoding='utf-8').read())
#     parser = FeatureChartParser(grammatik)
#
#     for baum in parser.parse(['drei', 'und', 'vier', 'zig']):
#         baum.draw()
#         term = baum.label()['SEM'].simplify()
#         print(term)          # -> plus(mal(1,3),mal(10,4))
#         print(wert(term))    # -> 43
#
# NLTK komponiert und beta-reduziert den SEM-Term, rechnet ihn aber nicht aus
# (nltk.sem.logic kennt keine Arithmetik). Diese Auswertung erledigt:
#
#     def wert(term):
#         return eval(str(term), {'plus': lambda a, b: a + b,
#                                 'mal':  lambda a, b: a * b})
#
# Zum Tokenisieren eignet sich ein laengster-Treffer-Zerleger ueber die unten
# aufgefuehrten Terminale.
#
# ---------------------------------------------------------------------------
# %(anzahl_regeln)d Regeln, %(anzahl_nt)d Nichtterminale, %(anzahl_terminale)d Terminale
# ---------------------------------------------------------------------------
"""


def _zahl_term(x):
    """Rundet einen Koeffizienten fuer den SEM-Term auf eine ganze Zahl.

    Die gelernten mappings kommen aus linearer Algebra und tragen Float-Rauschen
    (0.9999999999999998 statt 1). round() beseitigt es -- nltk.sem.logic koennte
    Dezimalpunkte ohnehin nicht parsen.
    """
    gerundet = int(round(x))
    return str(gerundet), gerundet


def _sem_term(mapping, variablen):
    """Baut den SEM-Term einer Regel aus mapping und den Variablen der Toechter.

    mapping = [a_0, ..., a_{n-1}, c] bedeutet Wert = sum(a_i * x_i) + c.
    Faktoren 1 und Summanden 0 werden weggelassen, damit die Terme lesbar
    bleiben (mal(1,?x) -> ?x).
    """
    summanden = []
    for i, var in enumerate(variablen):
        text, zahl = _zahl_term(mapping[i])
        if zahl == 0:
            continue
        summanden.append(str(var) if zahl == 1 else 'mal(%s,%s)' % (text, var))
    text, zahl = _zahl_term(mapping[-1])
    if zahl != 0 or not summanden:
        summanden.append(text)
    term = summanden[0]
    for weiterer in summanden[1:]:
        term = 'plus(%s,%s)' % (term, weiterer)
    return term


def _zerlege_cfgwort(cfgwort):
    """Zerlegt ein CFG-Wort in Terminal-/Nichtterminal-Stuecke.

    Im Format von scflist2cfg sind Grossbuchstaben Nichtterminale und
    Kleinbuchstaben Terminalzeichen. 'AundBzig' -> [('NT','A'), ('T','und'),
    ('NT','B'), ('T','zig')].
    """
    stuecke = []
    puffer = ''
    for zeichen in cfgwort:
        if zeichen.isupper():
            if puffer:
                stuecke.append(('T', puffer))
                puffer = ''
            stuecke.append(('NT', zeichen))
        else:
            puffer += zeichen
    if puffer:
        stuecke.append(('T', puffer))
    return stuecke


def cfg_nach_nltk_fcfg(cfg, dateiname='grammatik.fcfg'):
    """Uebersetzt die kategorisierte CFG in eine NLTK-Feature-Grammatik.

    Die Nichtterminale werden mit 'N' praefigiert (A -> NA), weil scflist2cfg
    'S' auch als regulaeres Nichtterminal vergibt und es sonst mit dem
    Startsymbol kollidieren wuerde.
    """
    def nt_name(buchstabe):
        return 'S' if buchstabe == 'S' else 'N' + buchstabe

    zeilen = []
    gesehen = set()
    nichtterminale = set()
    terminale = set()

    for regel in cfg:
        kopf, cfgwort, kategorie, mapping = regel[0], regel[1], regel[2], regel[3]
        stuecke = _zerlege_cfgwort(cfgwort)

        rechte_seite = []
        variablen = []
        for art, wert in stuecke:
            if art == 'NT':
                var = '?x%d' % len(variablen)
                variablen.append(var)
                name = nt_name(wert)
                nichtterminale.add(name)
                rechte_seite.append('%s[SEM=%s]' % (name, var))
            else:
                terminale.add(wert)
                rechte_seite.append("'%s'" % wert)

        if len(mapping) - 1 != len(variablen):
            # Sollte nicht vorkommen; lieber ueberspringen als eine kaputte
            # Regel schreiben.
            continue

        kopf_name = nt_name(kopf)
        nichtterminale.add(kopf_name)
        zeile = '%s[SEM=<%s>] -> %s' % (kopf_name, _sem_term(mapping, variablen),
                                        ' '.join(rechte_seite))
        if zeile in gesehen:
            continue
        gesehen.add(zeile)
        zeilen.append((kategorie, zeile))

    kopf_text = _NLTK_KOPF % {
        'dateiname': dateiname,
        'anzahl_regeln': len(zeilen),
        'anzahl_nt': len(nichtterminale),
        'anzahl_terminale': len(terminale),
    }

    # Nach Kategorie gruppieren -- c1 sind die kleinsten Bausteine.
    ausgabe = [kopf_text]
    for kategorie in sorted(set(k for k, _ in zeilen), key=lambda c: (len(c), c)):
        ausgabe.append('')
        ausgabe.append('# ---- Kategorie %s ----' % kategorie)
        for k, zeile in zeilen:
            if k == kategorie:
                ausgabe.append(zeile)
    ausgabe.append('')
    return '\n'.join(ausgabe)


# ===========================================================================
# Teil 4: Ausgabe-Einstiegspunkte fuer die GUI
# ===========================================================================

def _lexikon_als_liste(kenntnis_lexikon):
    """Eindeutige SCFunctions des Lexikons (ein Objekt kann unter mehreren
    Keys stehen)."""
    eindeutig = {}
    for eintrag in kenntnis_lexikon.values():
        eindeutig.setdefault(id(eintrag), eintrag)
    return list(eindeutig.values())


def erzeuge_cfg(kenntnis_lexikon, dateiname='grammatik.fcfg'):
    """Liefert (fcfg_text, anzahl_regeln). Ausgaben der Konversion werden
    abgefangen, damit sie nicht das Log der App fluten."""
    learnerlex = _lexikon_als_liste(kenntnis_lexikon)
    puffer, alt = io.StringIO(), sys.stdout
    sys.stdout = puffer
    try:
        cfg = scflist2cfg(learnerlex, kenntnis_lexikon=kenntnis_lexikon)
    finally:
        sys.stdout = alt
    return cfg_nach_nltk_fcfg(cfg, dateiname=dateiname), len(cfg)


def erzeuge_mg(kenntnis_lexikon):
    """Liefert (mg_text, anzahl_items) im unveraenderten Format von cfg2mg."""
    learnerlex = _lexikon_als_liste(kenntnis_lexikon)
    puffer, alt = io.StringIO(), sys.stdout
    sys.stdout = puffer
    try:
        cfg = scflist2cfg(learnerlex, kenntnis_lexikon=kenntnis_lexikon)
        mg = cfg2mg(cfg)
    finally:
        sys.stdout = alt

    zeilen = [
        'Minimalistische Grammatik (MG)',
        'Automatisch erzeugt aus dem gelernten Kenntnis-Lexikon.',
        '',
        'Format je Item:  [phonetische Form :: Merkmale], Semantik',
        '  =c   Selektor (waehlt eine Phrase der Kategorie c)',
        '  +X   Lizenzgeber (zieht eine Phrase mit -X an sich)',
        '  -X   Lizenznehmer',
        '',
        '%d Items' % len(mg),
        '-' * 70,
        '',
    ]
    zeilen += [item[0] for item in mg]
    zeilen.append('')
    return '\n'.join(zeilen), len(mg)
