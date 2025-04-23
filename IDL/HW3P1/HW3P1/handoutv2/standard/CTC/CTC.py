import numpy as np

class CTC(object):
    def __init__(self, BLANK=0):
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B, IY, IY, F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [BLANK, B, BLANK, IY, BLANK, IY, BLANK, F, BLANK]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0, 0, 0, 1, 0, 0, 0, 1, 0]
        """
        extended_symbols = [self.BLANK]
        skip_connect = [0]

        for i, symbol in enumerate(target):
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)
            
            if i > 0 and target[i] != target[i - 1]:
                skip_connect.append(1)
            else:
                skip_connect.append(0)
            
            skip_connect.append(0)

        extended_symbols = np.array(extended_symbols)
        skip_connect = np.array(skip_connect)

        return extended_symbols, skip_connect


    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros((T, S))

        alpha[0, 0] = logits[0, extended_symbols[0]]
        if S > 1:
            alpha[0, 1] = logits[0, extended_symbols[1]]

        for t in range(1, T):
            for i in range(S):
                if i > 0:
                    alpha[t, i] += alpha[t - 1, i - 1]
                alpha[t, i] += alpha[t - 1, i]
                if i > 1 and extended_symbols[i] != extended_symbols[i - 2] and skip_connect[i] == 1:
                    alpha[t, i] += alpha[t - 1, i - 2]
                alpha[t, i] *= logits[t, extended_symbols[i]]
        
        return alpha


    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
        """

        S = len(extended_symbols)
        T = len(logits)
        beta = np.zeros((T, S))

        beta[T - 1, S - 1] = 1
        if S > 1:
            beta[T - 1, S - 2] = 1

        for t in range(T - 2, -1, -1):
            for s in range(S):
                sym_idx = extended_symbols[s]

                a = beta[t + 1, s] * logits[t + 1, sym_idx]

                b = beta[t + 1, s + 1] * logits[t + 1, extended_symbols[s + 1]] if s + 1 < S else 0

                c = (
                    beta[t + 1, s + 2] * logits[t + 1, extended_symbols[s + 2]]
                    if s + 2 < S and skip_connect[s + 2] == 1
                    else 0
                )

                beta[t, s] = a + b + c

        return beta


    def get_posterior_probs(self, alpha, beta):
        T, S = alpha.shape
        gamma = np.zeros((T, S))

        for t in range(T):
            sumgamma = 0.0
            for i in range(S):
                gamma[t, i] = alpha[t, i] * beta[t, i]
                sumgamma += gamma[t, i]
            if sumgamma > 0:
                gamma[t] /= sumgamma
        
        return gamma


import numpy as np

class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

		Initialize instance variables

        Argument(s)
		-----------
		BLANK (int, optional): blank label index. Default 0.
        
		"""
		# -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
		# <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward"""

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []
        self.gammas = []

        for batch_idx in range(B):
            target_len = self.target_lengths[batch_idx]
            input_len = self.input_lengths[batch_idx]
            
            target_seq = target[batch_idx, :target_len]
            logits_seq = logits[:input_len, batch_idx, :]

            ext_symbols, skip_connect = self.ctc.extend_target_with_blank(target_seq)
            self.extended_symbols.append(ext_symbols)

            alpha = self.ctc.get_forward_probs(logits_seq, ext_symbols, skip_connect)

            beta = self.ctc.get_backward_probs(logits_seq, ext_symbols, skip_connect)

            gamma = self.ctc.get_posterior_probs(alpha, beta)
            self.gammas.append(gamma)

            loss = -np.sum(gamma * np.log(np.clip(logits_seq[:, ext_symbols], a_min=1e-10, a_max=None)))
            total_loss[batch_idx] = loss

        total_loss = np.sum(total_loss) / B

        return total_loss

    def backward(self):
        """
        CTC loss backward.
        """

        T, B, C = self.logits.shape
        dY = np.zeros_like(self.logits)

        for batch_idx in range(B):
            target_len = self.target_lengths[batch_idx]
            input_len = self.input_lengths[batch_idx]

            target_seq = self.target[batch_idx, :target_len]
            logits_seq = self.logits[:input_len, batch_idx, :]

            ext_symbols, skip_connect = self.ctc.extend_target_with_blank(target_seq)

            alpha = self.ctc.get_forward_probs(logits_seq, ext_symbols, skip_connect)
            beta = self.ctc.get_backward_probs(logits_seq, ext_symbols, skip_connect)

            gamma = self.ctc.get_posterior_probs(alpha, beta)

            for t in range(input_len):
                for s, symbol_idx in enumerate(ext_symbols):
                    dY[t, batch_idx, symbol_idx] -= gamma[t, s] / np.clip(logits_seq[t, symbol_idx], a_min=1e-10, a_max=None)

        return dY
