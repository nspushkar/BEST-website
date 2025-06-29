// Define all your code snippets here with languages
const codeSnippets = {
    full: {
        code: `% Family Relationships Knowledge Base

% Facts: Defining family relationships
parent(john, mary).
parent(mary, susan).
parent(mary, james).
parent(james, mark).
parent(anjali, kavya).

% Rule: Siblings are people who share the same parent
sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \\= Y.

% Rule: An ancestor is either a parent or someone who is a parent of an ancestor
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% Rule: Check if two people are related by common ancestors
related(X, Y) :- ancestor(Z, X), ancestor(Z, Y).

% Example Queries
% 1. Find siblings
% ?- sibling(mary, james).

% 2. Find ancestors
% ?- ancestor(john, mark).

% 3. Check if people are related
% ?- related(mary, susan).`,
        language: 'prolog'
    },
    
    facts: {
        code: `% Facts: Defining family relationships
parent(john, mary).
parent(mary, susan).
parent(mary, james).
parent(james, mark).
parent(anjali, kavya).`,
        language: 'prolog'
    },
    
    rules: {
        code: `% Rule: Siblings are people who share the same parent
sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \\= Y.

% Rule: An ancestor is either a parent or someone who is a parent of an ancestor
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% Rule: Check if two people are related by common ancestors
related(X, Y) :- ancestor(Z, X), ancestor(Z, Y).`,
        language: 'prolog'
    },
    
    queries: {
        code: `% Example Queries

% 1. Find siblings
?- sibling(mary, james).

% 2. Find ancestors
?- ancestor(john, mark).

% 3. Check if people are related
?- related(mary, susan).`,
        language: 'prolog'
    }
};

export default codeSnippets;
