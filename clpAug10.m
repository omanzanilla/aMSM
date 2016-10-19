function [W th cl Pf Nf]= clpAug10(solver,P,N,acc,basis)
%
% [W th cl Pf Nf] = clpAug10(solver,P,N)
% [W th cl Pf Nf] = clpAug10(solver,P,N,acc)
% [W th cl Pf Nf] = clpAug10(solver,P,N,acc)
% [W th cl Pf Nf] = clpAug10(solver,P,N,acc,basis)
% This function finds a piecewise nonlinear surface
% that separates  class P from class N.
%
%INPUT:
% -------------------------------------------------------
% Solver:
%    1. MATLAB linprog for 32 or 64 bits plattforms
%    2. COIN OR for 32 bits plattforms
%       To run this code you should have in PATH
%       clp.m, mexclp.cpp, mexclp.dll
%       Copyright (c) 2009, Thomas Trötscher
%       All rights reserved.
% -------------------------------------------------------
%
% P has as many rows as individuals of class P
% N has as many rows as individuals of class N
%    P(i,j) is the j-th feature of the i-th individual in P
%    N(i,j) is the j-th feature of the i-th individual in N
%
% 0 < acc <= 1   classification accuracy required
%                default acc = 1
%
%
% basis is a function handle or string containing the
% basis functions. If, for instance, basis = @gfun
% the user must provide 
% function g = gfun(x), which maps the feature space x 
%                       into the basis space
%
% If basis is omitted, the function finds a piecewise 
% linear surface; that is, g = x
%
%OUTPUT:
% n = length(g)
% t is the number of (non)linear pieces 
%   of the discriminating function
%
%    matrix  W    t columns, n rows
%    vector  th   t components
%    vector  cl   t components
%
% At the k-th iteration:
%   (hyper)plane W'(:,k) g(x) = th(k)
%    cl(k)  is the  class of classified individuals
%
% When there is no complete classification at exit:
%    Pf   unclassified elements in P in the basis space
%    Nf   unclassified elements in N in the basis space
%
% The code is based on the paper
% UM Garcia, O Manzanilla,
% A novel (non)linear piecewise classifier with a priori
% classification accuracy,
% Decision Support System DOI 10.1016/j.dss.2011.11.006
%
% This is a basic version. The author claims
% neither efficiency nor error free coding.
%
% Any comments are welcome
% 08 Aug 2010
% Ubaldo M. Garcia-Palomares

%Check input START
if (nargin < 3)
    error('Few input arguments: solver and classes P,N must be known')
end
if (nargin > 5)
    error('Many input arguments. Type help clpAug10')
end
[Prows, Pcols]= size(P);
[Nrows, Ncols]= size(N);

if (Prows == 0 | Pcols == 0)
    error('Class P is empty')
end
if (Nrows == 0 | Ncols == 0)
    error('Class N is empty')
end

if (Pcols ~= Ncols)
    error('Different number of features for P and N')
end

if ((Prows > 4000) && (Nrows > 4000))
    fprintf('Dataset too big for clp\n');
    fprintf('There are %d in P and %d in N\n',Prows,Nrows);
    error('At least one of them should be less than 4000');
end

if (nargin == 3)
    acc = 1;
end
if (acc <= 0 | acc > 1)
    error('acuracy must be in (0, 1]')
end

%Compute the basis space START
n= Pcols; %number of features
if (nargin== 5)
    g= feval(basis,P(1,:));
    n= length(g);  %Change # of features
    P(1,1:n)= g;
    for k= 2:Prows
        P(k,1:n)= feval(basis,P(k,1:Pcols));
    end
    for k= 1:Nrows
        N(k,1:n)= feval(basis,N(k,1:Ncols));
    end
    P= P(:,1:n);  N= N(:,1:n);
end
%Compute the basis space END

%At this point I have P, N in the g space

%Output values
W=[]; th=[]; cl=[];
Pf=[]; Nf=[];

IniPop= Prows + Nrows;   %Initial population
iter= 1;
NoLP= 0; % Program returns if 2 consecutive No LP solution
huge= 4000;  % This value is huge!!
big= huge;   % Program returns for big= 1

%Choosing initial class  START
if (Prows >= Nrows)
    clase= 1;    % clase stands for c(S) in paper
else
    clase= -1;
end
%Choosing initial class    END

while (Prows & Nrows)
   pop= Prows+ Nrows; %Actual population
   
   %To prevent working on HUGE LP models
   if (Prows> huge)
       clase= 1;
   elseif (Nrows> huge)
       clase= -1;
   end
   
   %    Choose S and X  START
   if (clase== 1)
       S= P;          % In this version 
       Srows= Prows;  % S = P
       X= N;
       Xrows= Nrows;
   else
       S= N;          % In this version 
       Srows= Nrows;  % S= N;
       X= P;
       Xrows= Prows;
   end
   %    Choose S and X  END
   
   % Size of problem  START
   if (Srows > big)
       % Keep Srows< big
       Srows= big;
       S= S(1:big,:);
   end
   % Size of problem    END
   
   %    Build A  START    Paper, section 3
   % Code model from paper START
   % model clp: Ax <= rhs = -1;
   
   if (clase== 1)
       A= [-S  ones(Srows,1) -eye(Srows);...
            X -ones(Xrows,1) zeros(Xrows,Srows)];
   else
       A= [ S -ones(Srows,1) -eye(Srows);...
           -X  ones(Xrows,1) zeros(Xrows,Srows)];
   end
   %    Build A  END  
   
   %    Build rhs  START
   rhs= -ones(Srows+Xrows,1);
   %    Build rhs  END
   
   % Srows is the #of components of y
   
   LPvar= n+1+Srows;  % # of LP variables in clp
   
   %    Objective START
   obj= ones(LPvar,1);   % To initialize!!!   
   obj(1:n+1)= 0;
   %    Objective END
   
   %    Lower and upper bound  START
   % -10 <= (w,th)  <= 10    B O U N D S
   lb= zeros(LPvar, 1);    % To initialize
   lb(1:n)= -10;  %     w lower bounds
   lb(n+1)= -100; % theta lower bound
   
   ub= 100*ones(LPvar, 1);
   ub(1:n)= 10;
   %    Lower and upper bound    END
   
   % I need to solve at least 2 LPs with
   % different constraints on w
     
   %    sum(w)    START
   sumw= zeros(1,LPvar);  %To initialize
   sumw(1:n)= 1; 
   %    sum(w)    END
   
   % Constraint sum(w) >= 1 for first  LP
   % Constraint sum(w) <=-1 for second LP
   
   % June 2011 Code for 64 bit plattforms  START
   % solver=1 linprog; solver= 2 clp
   if (solver== 1)
      [xp1 xd1 fail1]= linprog(obj,[A;-sumw],[rhs;-1],[],[],lb,ub);
      if (fail1> 0)
          fail1= 0;  % Optimum
      end
      [xp2 xd2 fail2]= linprog(obj,[A; sumw],[rhs;-1],[],[],lb,ub);
      if (fail2> 0)
          fail2= 0; % Optimum
      end
      
   elseif (solver== 2)
      [xp1 xd1 fail1]= clp([],obj,[A;-sumw],[rhs;-1],[],[],lb,ub);
      [xp2 xd2 fail2]= clp([],obj,[A; sumw],[rhs;-1],[],[],lb,ub);
   end
   % June 2011 Code for 64 bit plattforms  END
   
   % Coping with no LP solution   START
   if (fail1 & fail2)  %both LP fail
       fprintf('LP failure at iteration %d\n',iter);
       NoLP= NoLP+ 1;    % Another attempt
       if (NoLP== 2)    % Second attempt fails
           fprintf('Abnormal solution\n');
           Pf= P; Nf= N;
           return;  %Stronger than continue!!
       end
       
       %NEW July 14, 2010.   START
       %NEW July 14, 2010.   START

       % This artifice reduce the LP size
       %To solve medium datasets
       big= ceil(Srows/4);            %Reduce |S|
                            %In theory this WORKS
                     %for quadratic separating!!!
       %NEW July 14, 2010.     END
       
       % I am changing the class for the next iteration
       %    (just at the end of the WHILE loop)
       clase= -clase;
       
       continue;
   end
   % Coping with no LP solution     END
   
   % Continue with the algorithm
   
   % At this point at least one LP worked!!
   NoLP= 0;
   
   % Define the xp variables START
   if (fail1)
       xp1= xp2;
   elseif (fail2)
       xp2= xp1;
   end
   % Note that the variables are the same
   %    when a solution was LP failure
   % Define the xp variables  END
   
   % Solution with fewer unclassified START
   % Auxiliar hyperplanes (w1, th1) and (w2, th2)
   w1=  clase*xp1(1:n);
   th1= clase*xp1(n+1)- 1;
   w2=  clase*xp2(1:n);
   th2= clase*xp2(n+1)- 1;
   
   % Unclassified in the whole reduced set
   if (clase== 1)
       n1= (P* w1(:)- th1) < -1e-07; 
       n2= (P* w2(:)- th2) < -1e-07;
   else
       n1= (N* w1(:)- th1) < -1e-07; 
       n2= (N* w2(:)- th2) < -1e-07;
   end
   
   % Is Q empty?  START
   if (n1 & n2)   %OJO This is true if all components = 1
       %Q is empty
       fprintf('Q empty at iter. %d',iter);
       fprintf(' |S| = %d, |X| = %d\n', Srows, Xrows);
       if (Srows== 1)    % Consider a failure
           fprintf('Abnormal solution\n');
           Pf= P; Nf= N;
           return;  %Stronger than continue!!
       else
       
           %NEW July 14, 2010.   START
           %To solve medium datasets
           big= ceil(Srows/4);        %Reduce |S|
                            %In theory this WORKS
                     %for quadratic separation!!!
           %NEW July 14, 2010.     END
       end
       
       continue;
   end
   % Is Q empty?   END
   
   % Back to normal    START
   big= huge;
   % Back to normal    END
   
   cl(iter)= clase;
   
   % Choose the better solution MODIFIED JULY 15
   % Keep the better solution in w1,th1,n1
   if (sum(n2) < sum(n1))
       w1= w2; th1= th2; n1= n2;
   elseif (sum(n2)== sum(n1)...
          && sum(abs(w2))< sum(abs(w1)))
       w1= w2; th1= th2; n1= n2;
   end
   % Choose the better solution  END
   
    % Normalize the solution
    norma1= max(abs([w1(:); th1]));
    W(1:n,iter)= w1/norma1; 
    th(iter)= th1/norma1;

   %W(1:n,iter)= w1;
   %th(iter)= th1;
   
   % Reduced set.. n1 was defined in previous block
   if (clase== 1)
       P= P(n1,:);
       [Prows n]= size(P);
   else
       N= N(n1,:); 
       [Nrows n]= size(N);       
   end
   % Reduced set  END
   
   % Accuracy obtained ?
   if (sum(n1)/IniPop <= 1- acc)
       Pf= P; Nf= N;
       return;   % Done !!  Done !!
   end
   
   clase= -clase;
   iter= iter+ 1;  % next iteration
   
end  % while(Prows & Nrows)
   
return