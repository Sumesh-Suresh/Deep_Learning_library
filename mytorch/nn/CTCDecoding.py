import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1
        sym_set_with_blank = ['-'] + self.symbol_set

        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        for i in range(len(y_probs[0])):
            path_prob = path_prob * np.max(y_probs[:,i])
            index = np.argmax(y_probs[:, i])
            decoded_path.append(sym_set_with_blank[index])
        
        comp_decoded_path = ""
        for i in range(len(decoded_path)):
            if i==0 and decoded_path[i]!='-': comp_decoded_path += decoded_path[i]
            if decoded_path[i]=="-": continue
            if decoded_path[i]!=decoded_path[i-1] and decoded_path[i]!=comp_decoded_path[-1]: comp_decoded_path += decoded_path[i]

        return comp_decoded_path, path_prob
        # raise NotImplementedError


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width
    
    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None
        PathScore, BlankPathScore = dict(), dict()
        
        def InitializePaths(SymbolSet, y):
            InitialBlankPathScore, InitialPathScore = dict(), dict()
            path = ""
            InitialBlankPathScore[path] = y[0] # Score of blank at t=1 
            InitialPathsWithFinalBlank = {path}
            # Push rest of the symbols into a path-ending-with-symbol stack
            InitialPathsWithFinalSymbol = set()
            for i in range(len(SymbolSet)): # This is the entire symbol set, without the blank
                InitialPathScore[SymbolSet[i]] = y[i+1] # Score of symbol c at t=1 
                InitialPathsWithFinalSymbol.add(SymbolSet[i]) # Set addition
            
            return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

        def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
            UpdatedPathsWithTerminalBlank = set()
            UpdatedBlankPathScore = dict()
            for path in PathsWithTerminalBlank:
                if path not in UpdatedPathsWithTerminalBlank:
                    UpdatedPathsWithTerminalBlank.add(path) # Set addition 
                    UpdatedBlankPathScore[path] = BlankPathScore[path]*y[0]

            for path in PathsWithTerminalSymbol:
                if path in UpdatedPathsWithTerminalBlank:
                    UpdatedBlankPathScore[path] += PathScore[path]* y[0]
                else:
                    UpdatedPathsWithTerminalBlank.add(path) # Set addition
                    UpdatedBlankPathScore[path] = PathScore[path] * y[0]
                    
            return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

        def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y):
            UpdatedPathsWithTerminalSymbol = set()
            UpdatedPathScore = dict()
            for path in PathsWithTerminalBlank:
                for i, c in enumerate(SymbolSet): # SymbolSet does not include blanks
                    newpath = path + c # Concatenation 
                    UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition 
                    UpdatedPathScore[newpath] = BlankPathScore[path] * y[i+1]
            
            for path in PathsWithTerminalSymbol:
                for i, c in enumerate(SymbolSet): # SymbolSet does not include blanks
                    if (c == path[-1]):
                        newpath = path 
                    else: 
                        newpath = path + c # Horizontal transitions don’t extend the sequence 
                    if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                        UpdatedPathScore[newpath] += PathScore[path] * y[i+1]
                    else:# Create new path
                        UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition 
                        UpdatedPathScore[newpath] = PathScore[path] * y[i+1]
            
            return UpdatedPathsWithTerminalSymbol, UpdatedPathScore
        
        def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
            PrunedBlankPathScore = dict()
            PrunedPathScore = dict()
            scorelist = []

            for p in PathsWithTerminalBlank:
                scorelist.append(BlankPathScore[p])
            
            for p in PathsWithTerminalSymbol:
                scorelist.append(PathScore[p])
          
            scorelist = sorted(scorelist)[::-1] # In decreasing order
            if BeamWidth < len(scorelist):
                cutoff = scorelist[BeamWidth] 
            else: 
                cutoff = scorelist[-1]

            PrunedPathsWithTerminalBlank = set()
            for p in PathsWithTerminalBlank:
                if BlankPathScore[p] > cutoff :
                    PrunedPathsWithTerminalBlank.add(p) # Set addition 
                    PrunedBlankPathScore[p] = BlankPathScore[p]

            PrunedPathsWithTerminalSymbol = set()
            for p in PathsWithTerminalSymbol:
                if PathScore[p] > cutoff :
                    PrunedPathsWithTerminalSymbol.add(p) # Set addition 
                    PrunedPathScore[p] = PathScore[p]
            
            return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore


        def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
            MergedPaths = PathsWithTerminalSymbol 
            FinalPathScore = PathScore
            for p in PathsWithTerminalBlank:
                if p in MergedPaths:
                    FinalPathScore[p] += BlankPathScore[p]
                else:
                    MergedPaths.add(p) # Set addition
                    FinalPathScore[p] = BlankPathScore[p]
            
            return MergedPaths, FinalPathScore
        
        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(self.symbol_set, y_probs[:,0])

        for t in  range(1,T):
            PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, self.beam_width)
            NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:,t])
            NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, self.symbol_set, y_probs[:,t])
       
        MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
        bestPath, _ = sorted(FinalPathScore.items(), key=lambda x: x[1])[-1]            
        
        return bestPath, FinalPathScore
