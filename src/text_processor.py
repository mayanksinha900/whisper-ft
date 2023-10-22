import re
from typing import List, Optional

from tqdm import tqdm


class TextPreprocess:
    """
    apply_global: bool = True 
    > For setting whether the keyword used to replace is global or not
    > If set false, you need to specify replace keyword for every function
    > Cannot use pipeline or pipeline_batch function if set apply_global is set False

    replace_keyword: str = " "
    > Default is to replace with withspaces.

    verbose: bool =  True
    > Whether to show progress or not
    > Only used when using the function pipeline_batch
    """
    def __init__(self, 
                 apply_global:bool = True, 
                 replace_keyword:str = ' ', 
                 verbose:bool = True, 
                 only_en:bool = False):
        
        self.apply_global = apply_global
        self.replace_keyword = replace_keyword
        self.verbose = verbose
        self.only_en = only_en

    def remove_url(self, text:str, replace: Optional[str] = None) -> str:
        if self.apply_global:
            text = re.sub(r'http\S+', self.replace_keyword, text)
            text = re.sub(r'www\S+', self.replace_keyword, text)
        else:
            assert replace is not None
            text = re.sub(r'http\S+', replace, text)
            text = re.sub(r'www\S+', replace, text)
        return text

    def remove_mentions(self, text:str, replace: Optional[str] = None) -> str:
        if self.apply_global:
            text = re.sub(r'@\S+ ', self.replace_keyword, text)
            text = re.sub(r'#\S+ ', self.replace_keyword, text)
        else:
            assert replace is not None
            text = re.sub(r'@\S+ ', replace, text)
            text = re.sub(r'#\S+ ', replace, text)
        return text

    def remove_num(self, text:str, replace: Optional[str] = None) -> str:
        if self.apply_global:
            text = re.sub(r'[0-9]', self.replace_keyword, text)
        else:
            assert replace is not None
            text = re.sub(r'[0-9]', replace, text)
        return text

    def remove_special_chars(self, text:str, replace: Optional[str] = None) -> str:
        reg = """
            ^`~!@#$%^&*()_+={}|\\:;"'<,>.?๐฿]*$
            """
        regex = r"[()|/,.;:%$@!%^₹›‹\"\'\\\[\]{}&?“”—’*“©،#–+=،؟-]"
        if self.apply_global:
            text = re.sub(regex, self.replace_keyword, text)
            text = re.sub(reg, self.replace_keyword, text)
            text = re.sub(r'\n', self.replace_keyword, text)
            text = re.sub(r'\t', self.replace_keyword, text)
            text = re.sub(r'\r', self.replace_keyword, text)
            text = re.sub(r' +', self.replace_keyword, text)
        else:
            assert replace is not None
            text = re.sub(regex, replace, text)
            text = re.sub(reg, replace, text)
            text = re.sub(r'\n', replace, text)
            text = re.sub(r'\t', replace, text)
            text = re.sub(r'\r', replace, text)
            text = re.sub(r' +', replace, text)
        return text
    
    def keep_only_en(self, text:str, replace: Optional[str] = None) -> str:
        if self.apply_global:
            text = re.sub(r"[^A-Za-z]", self.replace_keyword, text)
        else:
            text = re.sub(r"[^A-Za-z]", replace, text)

        return text

    def pipeline(self, text:str) -> str:
        assert self.apply_global is True
        text = str(text).lower()
        text = self.remove_mentions(text)
        text = self.remove_url(text)
        text = self.remove_num(text)
        text = self.remove_special_chars(text)
        if self.only_en:
            text = self.keep_only_en(text)
            text = re.sub(r' +', self.replace_keyword, text)

        return text

    def pipeline_batch(self, texts: List[str]) -> List[str]:
        assert self.apply_global is True
        temp = []
        if self.verbose:
            for text in tqdm(texts):
                temp.append(self.pipeline(text))
        else:
            for text in texts:
                temp.append(self.pipeline(text))
