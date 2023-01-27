class Solution(object):
    def strStr(self, haystack, needle):
        _next = self.get_next(needle)
        h_pt, n_pt = 0, 0
        
        while h_pt < len(haystack):
            if haystack[h_pt] == needle[n_pt]:
                h_pt += 1
                n_pt += 1
                
                if n_pt == len(needle): return True
            
            else:
                if n_pt == 0:
                    h_pt += 1
                else:
                    n_pt = _next[n_pt - 1]
        
        return False
    
    def get_next(self, s):
        _next = [0] * len(s)
        i, j = 1, 0
        
        while i < len(s):
            
            while s[i] != s[j] and j != 0:
                j = _next[j - 1]
            
            if s[i] == s[j]:
                j += 1
            
            _next[i] = j
            i += 1
        
        return _next

print(Solution().strStr("mississippi", "issip"))