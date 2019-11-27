stimulus_situation_verbal(papers).
stimulus_situation_verbal(manuals).
stimulus_situation_verbal(documents).
stimulus_situation_verbal(textbooks).

stimulus_situation_visual(pictures).
stimulus_situation_visual(illustrations).
stimulus_situation_visual(photographs).
stimulus_situation_visual(daigrams).

stimulus_situation_physical(machines).
stimulus_situation_physical(buildings).
stimulus_situation_physical(tools).

stimulus_situation_symbolic(numbers).
stimulus_situation_symbolic(formulas).
stimulus_situation_symbolic('computer programs').


stimulus_response_oral(lecturing).
stimulus_response_oral(advising).
stimulus_response_oral(counselling).


stimulus_response_hands_on(building).
stimulus_response_hands_on(repairing).
stimulus_response_hands_on(troubleshooting).

stimulus_response_documented(writing).
stimulus_response_documented(typing).
stimulus_response_documented(drawing).

stimulus_response_analytical(evaluating).
stimulus_response_analytical(reasoning).
stimulus_response_analytical(investigating).

feedback(required).

workshop(X,Y,Z):-stimulus_situation_physical(X),stimulus_response_hands_on(Y),feedback(Z).
lecture_tutorial(X,Y,Z):-stimulus_situation_symbolic(X),stimulus_response_analytical(Y),feedback(Z);stimulus_situation_visual(X),stimulus_response_oral(Y),feedback(Z);stimulus_situation_verbal(X),stimulus_response_analytical(Y),feedback(Z).
videocassette(X,Y,Z):-stimulus_situation_visual(X),stimulus_response_documented(Y),not(feedback(Z)).
role_play_exercise(X,Y,Z):-stimulus_situation_verbal(X),stimulus_response_oral(Y),feedback(Z).

medium(X,Y,Z):-(workshop(X,Y,Z)->writeln('workshop'));(lecture_tutorial(X,Y,Z)->writeln('lecture tutorial'));(videocassette(X,Y,Z)->writeln('videocasstte'));(role_play_exercise(X,Y,Z)->writeln('role play exercise')).
