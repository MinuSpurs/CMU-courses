import numpy as np

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
        Perform greedy search decoding.
        """
        decoded_path = []
        blank_index = 0
        path_prob = 1.0

        _, seq_len, batch_size = y_probs.shape

        for time_step in range(seq_len):
            max_symbol_index = np.argmax(y_probs[:, time_step, 0])
            max_symbol_prob = y_probs[max_symbol_index, time_step, 0]
            path_prob *= max_symbol_prob

            if max_symbol_index != blank_index:
                if not decoded_path or decoded_path[-1] != max_symbol_index:
                    decoded_path.append(max_symbol_index)

        decoded_path = ''.join(self.symbol_set[idx - 1] for idx in decoded_path)

        return decoded_path, path_prob


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
        blank_index = 0  

        bestPaths = {(): 1.0}

        for t in range(T):
            TempbestPaths = {}  
            for path, score in bestPaths.items():
                for s_index in range(len(self.symbol_set) + 1):
                    s_prob = y_probs[s_index, t, 0] 
                    new_score = score * s_prob 

                    if s_index == blank_index:
                        s = "-" 
                    else:
                        s = self.symbol_set[s_index - 1]  

                    if len(path) == 0:
                        new_path = path + (s,)
                    else:
                        if path[-1] == "-":
                            if s == "-":
                                new_path = path  
                            elif len(path) == 1:
                                new_path = (s,)  
                            else:
                                new_path = path[:-1] + (s,) 
                        else:
                            if s == "-":
                                new_path = path + (s,) 
                            elif s == path[-1]:
                                new_path = path  
                            else:
                                new_path = path + (s,)  

                    TempbestPaths[new_path] = TempbestPaths.get(new_path, 0) + new_score

            sorted_paths = sorted(TempbestPaths.items(), key=lambda x: x[1], reverse=True)
            bestPaths = dict(sorted_paths[: self.beam_width])

        FinalPathScore = {}
        for path, score in TempbestPaths.items():
            score = np.array([score])
            if path[-1] == "-":
                path = path[:-1] 
            path_str = "".join([c for c in path if c != "-"])  
            FinalPathScore[path_str] = FinalPathScore.get(path_str, 0) + score

        FinalPathScore = dict(sorted(FinalPathScore.items(), key=lambda x: x[1], reverse=True))

        bestPath = max(FinalPathScore.items(), key=lambda x: x[1])[0]

        return bestPath, FinalPathScore


