import numpy as np
import matplotlib.pyplot as plt
import tqdm

class Question():
    def __init__(self, confusion = 0.0, uncertainty = 0.0):
        self.confusion = confusion 
        self.uncertainty = uncertainty
        # confusion is a float between 0 and 1, 0 indicates no confounding answers exist, 1 indicates a single confounding answer dominates the correct answer
        # uncertainty is a float between 0 and 1, 0 indicates that a model presented with this question will always answer consistently, 1 indicates that a model presented with this question will always answer randomly
    
class Exam():
    def __init__(self, n_questions, confusions, uncertainties):
        self.n_questions = n_questions
        self.confusions = confusions
        self.uncertainties = uncertainties
        
        if isinstance(self.confusions, (int, float)):
            self.confusions = [self.confusions for i in range(n_questions)]
        if isinstance(self.uncertainties, (int, float)):
            self.uncertainties = [self.uncertainties for i in range(n_questions)]
        
        assert len(self.confusions) == n_questions, 'provided confusions list must have the same length as the number of questions'
        assert len(self.uncertainties) == n_questions, 'provided uncertainties list must have the same length as the number of questions'
        
        self.questions = [Question(d[0],d[1]) for d in zip(self.confusions, self.uncertainties)]
            
class Agent():
    def __init__(self, competence = 1):
        self.competence = competence
        # competence is a float between 0 and 1, 0 indicates the model responds completely randomly, 1 indicates the model responds in line with its belief 100% of the time

    def answer(self, question):
        adjusted_uncertainty = min(1, question.uncertainty / self.competence)
        adjusted_confusion = min(1, question.confusion / self.competence)
        return (np.random.binomial(p=1 - adjusted_confusion, n=1, ) * 2 - 1) * np.random.binomial(p=1 - adjusted_uncertainty, n=1, ) # 1 if correct, -1 if confused, 0 if stochastic
    
    def take_exam(self, exam):
        return [self.answer(q) for q in exam.questions]
            
class Ensemble():
    def __init__(self, n_agents, competencies):
        self.n_agents = n_agents
        self.competencies = competencies
        
        if isinstance(self.competencies, (int, float)):
            self.agents = [Agent(competencies) for i in range(n_agents)]
        elif type(self.competencies) == list or type(self.competencies) == np.ndarray:
            assert len(self.competencies) == n_agents, 'provided competencies list must have the same length as the number of agents'
            self.agents = [Agent(c) for c in self.competencies]
        else:
            assert False, 'competencies must be either an number, a list or a numpy array'
                    
    def take_exam(self, exam):
        return [a.take_exam(exam) for a in self.agents]
    
    def vote(self, answers, threshold):
        answers = np.array(answers)
        consensus = np.mean(np.where(answers != 0, 1, 0), axis=0) >= threshold
        responses = np.where(np.mean(answers, axis=0) > 0, 1, 0)
        responses = np.where(np.mean(answers, axis=0) < 0, -1, responses)
        return responses * consensus
    
    def vote_count(self, responses):
        return [np.sum(np.array(responses) == i) for i in [-1, 0, 1]]
        