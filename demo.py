import numpy as np
import opencv-python-headless as cv2
import hashlib
import scipy.stats as st

private_key=b"abcd1234"

def remove_special(s):
    # Remove all non-special characters from text
    ret_s=""
    for char in s:
        if (ord(char) >= ord('a') and ord(char) <= ord('z')) or char==" ":     
            ret_s+=char
    return ret_s

class BaseText():
    # Base module with some crypto functions
    def __init__(self):
        valid_chars="abcdefghijklmnopqrstuvwxyz?.!-,:;\'\"_ABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890 "
        self.valid_chars=[*valid_chars]
        alphabet="abcdefghijklmnopqrstuvwxyz"
        self.alphabet=[*alphabet]
        self.chars_in_bl=1 # how many characters in blacklist when we generate it
        self.chars_for_bl=3 # how many characters from start of word do we apply a blacklist to?
    
    def blacklist(self,seed_phrase):
        # Takes a seed_phase (a byte string), seeds numpy with it, chooses number of letters in blacklist, then returns the blacklist
        bl=[]
        # Seed numpy RNG - we get maximum digest that doesn't overflow numpy manual seeding
        h=hashlib.shake_256(seed_phrase)
        numpy_seed=int(h.hexdigest(4),16)
        np.random.seed(numpy_seed)
        bl_index=np.random.choice(len(self.alphabet),self.chars_in_bl,replace=False)
        for idx in bl_index:
            bl.append(self.alphabet[idx])
            bl.append(self.alphabet[idx].upper())
        return bl 
    
    def get_encoding_string(self,text):
        # Takes an input text, and returns the current encoding string for that text (basically takes last 2 words, plus current word so far)
        # Full stops should reset the encoding
        # If two letters or more in latest word, return blank as we only blacklist the first two characters
        txt=text.replace("?",".").replace("!",".")
        txts=txt.split(".")
        txt=remove_special(txts[-1].lstrip().lower())
        words=txt.split(" ")

        if len(words[-1])>=self.chars_for_bl:
            encoding_string=b""
        else:
            encoding_string="".join(words[-3:]).encode("utf-8")
        return encoding_string
    
    def z_score(self,text):
        # Goes over text, counts number of blacklisted chars, uses intensity to evaluate a z score
        words=text.split(" ")
        count=0
        count -=1 # first letter of text is free generated
        txt=text.replace("?",".").replace("!",".")
        count-= txt.count(".") # first in new sentence is free
        x=0 # can accumulate x as number of blacklisted if we allow any blacklisted chars. Otherwise, it is just 0 as it is a hard blacklist
        # We use the normal approx to binomial, as opposed to direct probability calculation, as it is more flexible for when we allow x>0
        for word in words:
            count += min(len(word),self.chars_for_bl)
        if count<=0:
            return float("inf")
        p=self.chars_in_bl/26
        mu=count*p
        sigma=np.sqrt(count*p*(1-p))
        z_score=(x-mu)/sigma
        return z_score

        

class Writer(BaseText):
    def __init__(self,key):
        super().__init__()
        self.pk=key
        self.text=""
        self.encoding_str=""
        self.height = 800
        self.width = 1200
        self.bl=[]
        self.display()

    def run(self):       
        key = cv2.waitKeyEx(0)
        if key == 27:
            return False
        if key==8:
            self.text=self.text[:-1]
        # Check if key is valid
        if key in [ord(char) for char in self.valid_chars]:
            # Check if key blacklisted
            char=chr(key)
            if char in self.bl:
                # We can either put it in the text and highlight it, or ignore it
                pass
            else:
                self.text+=char
        
        # Generate blacklist for next character
        seed_phrase=self.get_encoding_string(self.text)

        # Blank seed_phrase should return empty blacklist
        if seed_phrase==b"":
            self.bl=[]
        # Otherwise get the blacklist
        else:
            seed_phrase=self.pk+seed_phrase
            self.bl=self.blacklist(seed_phrase)
        self.display()
        return True

    def display(self):
        display_text=self.text.split(" ")
        result = np.ones((self.height, self.width, 3), np.uint8)*255
    
        FONT_SCALE = 0.6
        FONT = cv2.FONT_HERSHEY_TRIPLEX
        FONT_THICKNESS = 1
        word_x, word_y = 40, 120  # Origin cordinate of text
        height_spacing = 10  # space between 2 string

        for word in display_text:
            (label_width, label_height), _ = cv2.getTextSize(word + " ", FONT, FONT_SCALE, FONT_THICKNESS)

            if word_x + label_width > self.width:
                word_x = 20
                word_y += label_height + height_spacing

            cv2.putText(result, word + " ", (word_x, word_y), FONT, FONT_SCALE, (0,0,0), FONT_THICKNESS)
            word_x += label_width
        z_score=self.z_score(self.text)
        prob=st.norm.cdf(z_score) # prob of model generation
        cv2.putText(result,f"BLACKLIST: ",(20,50),FONT,1,255,1)
        cv2.putText(result,f"{', '.join(self.bl[1::2])}",(220,50),FONT,1,(0,0,0),2)
        cv2.putText(result,f"Close with Escape",(800,750),FONT,0.6,(0,0,255),1)
        cv2.putText(result,f"Model generated probability: {prob:.4f}",(100,750),FONT,0.6,255,1)
        cv2.imshow("",result)
    



if __name__ == '__main__':
    W=Writer(private_key)
    while True:
        val=W.run()
        if not val:
            break
    cv2.destroyAllWindows()
