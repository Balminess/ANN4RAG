import jsonlines
import re
import editdistance
import re
from difflib import SequenceMatcher
from suffix_trees import STree

class Evaluation:
    def __init__(self, expected: str, actual: str):
        self.expected_orig = expected.strip()
        self.actual_orig = actual.strip()

        self.expected = expected.strip()
        self.actual = actual.strip()
    
    def is_exact_match(self):
        return self.expected== self.actual

    def _strip_full_form(self, assertion):
        listical = list(assertion.split())
        final_str = ''
        found_assertion = False
        for i in range(len(listical)):
            if listical[i].startswith('assert'):
                found_assertion = True
            if found_assertion:
                final_str += listical[i] + ' '
        return final_str

    def _strip_extra_parenthesis(self):

        if '(' in self.expected and ')' in self.expected:
            self.expected = self.expected.replace('(', '')
            self.expected = self.expected.replace(')', '')


    def _replace_assert_true_false_assert_equal(self):
        ASSERT_EQUALS_TRUE = 'assertEquals(true,'
        ASSERT_EQUALS_FALSE = 'assertEquals(false,'
        ASSERT_TRUE = 'assertTrue('
        ASSERT_FALSE = 'assertFalse('
        if (ASSERT_EQUALS_TRUE in self.expected and ASSERT_TRUE in self.actual) or \
                ASSERT_EQUALS_TRUE in self.actual and ASSERT_TRUE in self.expected:
            self.expected = self.expected.replace(ASSERT_EQUALS_TRUE, ASSERT_TRUE)
            self.actual = self.actual.replace(ASSERT_EQUALS_TRUE, ASSERT_TRUE)
        elif (ASSERT_EQUALS_FALSE in self.expected and ASSERT_FALSE in self.actual) or \
                ASSERT_EQUALS_FALSE in self.actual and ASSERT_FALSE in self.expected:
            self.expected = self.expected.replace(ASSERT_EQUALS_FALSE, ASSERT_FALSE)
            self.actual = self.actual.replace(ASSERT_EQUALS_FALSE, ASSERT_FALSE)

    def _match_args(self):
        def find_match(text):
            x = re.findall("\(\s*([^)]+?)\s*\)", text)
            if len(x):
                return [a.strip() for a in x[0].split(',')]
            return []

        def get_assertion_type(text):
            for c in text.split():
                if c.startswith('assert'):
                    return c

        expected_args = sorted(find_match(self.expected))
        actual_args = sorted(find_match(self.actual))

        expected_assertion_type = get_assertion_type(self.expected)
        actual_assertion_type = get_assertion_type(self.actual)
        return len(expected_args) and len(actual_args) and \
            expected_args == actual_args and expected_assertion_type == actual_assertion_type #参数一样且类型一致

    def is_match(self):
        if self.expected == self.actual:
            return True

        if self._match_args():
            return True
        
        self._replace_assert_true_false_assert_equal()
        if self.expected == self.actual:
            return True

        return False

    def calc_lcs(self):
        """
        https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
        """
        try:
            input_list = [self.expected_orig, self.actual_orig]
            st = STree.STree(input_list)
            longest_lcs = st.lcs()
        except RecursionError as e:
            print(e)
            print(f"error in calc_lcs for {self.expected_orig} and {self.actual_orig}")
            match = SequenceMatcher(None, self.expected_orig, self.actual_orig)\
                .find_longest_match(0, len(self.expected_orig), 0, len(self.actual_orig))
            longest_lcs = self.expected_orig[match.a:match.a + match.size]


        return longest_lcs

    def edit_distance(self):
        return editdistance.eval(self.expected_orig, self.actual_orig)

    def calculate(self):
        return {
            "is_exact_match": self.is_exact_match(),
            "is_match": self.is_match(),
            "calc_lcs": self.calc_lcs(),
            "edit_distance": self.edit_distance()
        }

def extract_info(text):
    pattern = r'org'
    start = re.search(pattern, text)
    
    if not start:
        return text
    
    start_index = start.start()
    count = 0
    end_index = start_index
    
    for i in range(start_index, len(text)):
        if text[i] == '(':
            count += 1
        elif text[i] == ')':
            count -= 1
            if count == 0:
                end_index = i
                break
    
    return text[start_index:end_index + 1]

def caculate_atlas(file_path):
    data = []

    with jsonlines.open(file_path) as f: 
        for i in f:   
            data.append(i)
    ASSERTION_TYPES = ["assertEquals", "assertTrue", "assertNotNull",
                    "assertThat", "assertNull", "assertFalse",
                    "assertArrayEquals", "assertSame"]

    exact_match_count = 0
    match_count = 0
    assertion_type_matched_count = 0
    total_count = 0
    sum_lcs_percentage = 0
    sum_edit_distance = 0

    true_index=[]

    for i in data:
   
        expect=i['label'].replace(" ", "").replace(";", "")
        ourstr=extract_info(i['actual'])
        actual=ourstr.replace(" ", "").replace(";", "")


        result=Evaluation(expect, actual).calculate()
        is_exact_match=result["is_exact_match"]
        is_match=result["is_match"]
        lcs=result["calc_lcs"]
        edit_distance=result["edit_distance"]

        if len(lcs) != 0:
            sum_lcs_percentage += (len(lcs) /len(actual) ) * 100
                
        expected_assertion_type = None
        actual_assertion_type = None
        for a in ASSERTION_TYPES:
            if a in i['label']:
                expected_assertion_type = a
            if a in i['actual']:
                actual_assertion_type = a
        
        assertion_type_matched = expected_assertion_type == actual_assertion_type

        if assertion_type_matched:
            assertion_type_matched_count += 1

        sum_edit_distance += int(edit_distance)
        if is_exact_match:
            exact_match_count = exact_match_count + 1

            true_index.append(i["idx"])
            
        if is_match:
            match_count = match_count + 1
        total_count = total_count + 1
            
    avg_lcs_percentage = sum_lcs_percentage/total_count
    avg_edit_distance = sum_edit_distance/total_count

    return exact_match_count/total_count,avg_lcs_percentage,match_count/total_count,avg_edit_distance,assertion_type_matched_count / total_count