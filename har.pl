nn(net1, [X], Y, [yes, no]) :: mv_lower_leg(X,Y).
nn(net2, [X], Y, [yes, no]) :: mv_lower_leg_alot(X,Y).
nn(net3, [X], Y, [yes, no]) :: mv_right_arm(X,Y).
nn(net4, [X], Y, [yes, no]) :: mv_upper_leg(X,Y).
nn(net5, [X], Y, [yes, no]) :: mv_left_arm(X,Y).
nn(net6, [X], Y, [yes, no]) :: mv_forearm(X,Y).

activity(X,walk)  :- mv_lower_leg(X,yes), mv_lower_leg_alot(X,yes), mv_right_arm(X,no).
activity(X,run)   :- mv_lower_leg(X,yes), mv_lower_leg_alot(X,yes), mv_right_arm(X,yes).
activity(X,squat) :- mv_lower_leg(X,yes), mv_lower_leg_alot(X,no), mv_upper_leg(X,yes).
activity(X,jump)  :- mv_lower_leg(X,yes), mv_lower_leg_alot(X,no), mv_upper_leg(X,no).
activity(X,wave)  :- mv_lower_leg(X,no),  mv_left_arm(X,yes).
activity(X,clap)  :- mv_lower_leg(X,no),  mv_left_arm(X,no), mv_forearm(X,yes).
activity(X,wipe)  :- mv_lower_leg(X,no),  mv_left_arm(X,no), mv_forearm(X,no).