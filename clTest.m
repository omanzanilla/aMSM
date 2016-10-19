 % Test different databases with the classification
 % algorithm proposed by Garcia and Manzanilla
 % See code clpAug10.m
 %
 % You can test your own dataset. Please provide
 % ascii files:
 %      N_user.txt  for the N class,
 %      P_user.txt  for the P class.
 
 solver= menu('Choose the solver',...
           'MATLAB linprog for 32 or 64 bit plattforms',...
           'clp for 32 bit plattforms');
 switch solver
     case 1
         disp('The solver is MatLab linprog');
     case 2
         disp('The solver is COIN-OR clp');
 end

 d= ' database.';
 k= menu('Choose a database','abalone','bank','cancer',...
     'contraceptive','credit','data','diabetes',...
     'diabetesnew','heart','housing','ionosphere','letter',...
     'parabola','sonar','spiral','synthetic','WineQuality',...
     'user provides P_user.txt, N_user.txt');
 
 switch k
     case 1
         database='abalone';
         disp([database d])
         N= load('abalone.txt');
         [Nrows n]= size(N);
         
         ka= menu('Age to identify','infant','young','old');
         id= 'Identifying ';
         switch ka
             case 1
                 disp([id 'infant.'])
                 M= N(:,n) < 9;
             case 2
                 disp([id 'young'])
                 M= (N(:,n)== 10) | (N(:,n)== 9);
             case 3
                 disp([id 'old'])
                 M= N(:,n)> 10;
         end
         P= N(M,1:n-1); N= N(~M,1:n-1);
         
     case 2
         database= 'bank';
         P= load('P_bank.txt');
         N= load('N_bank.txt');
         disp([database d])

     case 3
         database= 'cancer';
         P= load('P_BCancerW.txt');
         N= load('N_BCancerW.txt');
         disp([database d])
         
     case 4
         database= 'contraceptive';
         P= load('P_contraceptive.txt');
         N= load('N_contraceptive.txt');
         disp([database d])
         
     case 5
         database= 'credit';
         P= load('P_credit.txt');
         N= load('N_credit.txt');
         disp([database d])
         
     case 6
         database= 'data';
         P= load('P_data.txt');
         N= load('N_data.txt');
         disp([database d])
         
     case 7
         database= 'diabetes';
         P= load('P_Diabetes.txt');
         N= load('N_Diabetes.txt');
         disp([database d])
         
     case 8
         database= 'diabetes new';
         P= load('P_DiabetesNew.txt');
         N= load('N_DiabetesNew.txt');
         disp([database d])
         
     case 9
         database= 'heart';
         P= load('P_Heart.txt');
         N= load('N_Heart.txt');
         disp([database d])
         
     case 10
         database= 'housing';
         disp([database d])
         price= input('State the house price (Commonly = 21): ');
         if isempty(price) 
             price= 21;
         end
         House= load('housing.txt');
         komp= find(House(:,14) >= price);
         P= House(komp,1:13);
         komp= find(House(:,14) < price);
         N= House(komp,1:13);         
         
     case 11
         database= 'ionosphere';
         P= load('P_ionosphere.txt');
         N= load('N_ionosphere.txt');
         disp([database d])
         
     case 12   % Less than 6000 individuals
         database= 'letter';
         disp([database d])
         
         alphabet= ['ABCDEFGHIJKLMNOPQRSTUVWXYZ'];
         ka= menu('Letter to recognize','A','B','C','D','E',...
             'F','G','H','I','J','K','L','M','N','O','P',...
             'Q','R','S','T','U','V','W','X','Y','Z');
         letra= alphabet(ka);
         disp(['Identifying letter ' letra])
         
         P= []; N= []; tam= 0;
         fa= fopen('letter.txt');
         while ~feof(fa) %& tam< 5000
             line= fgets(fa);
             if strcmp(line(1),letra)
                 P= [P; str2num(line(3:end))];
             else
                 N= [N; str2num(line(3:end))];
             end
             [tam n]= size(N);
         end
         fclose(fa);
                      
     case 13
         database= 'parabola';
         P= load('P_parabola.txt');
         N= load('N_parabola.txt');
         disp([database d])
         
     case 14
         database= 'sonar';
         P= load('P_sonar.txt');
         N= load('N_sonar.txt');
         disp([database d])
         
     case 15
         database= 'spiral';
         P= load('P_spiral2x2.txt');
         N= load('N_spiral2x2.txt');
         disp([database d])
         
     case 16
         database= 'synthetic';
         disp([database d])
         cor= menu('D A T A  corrupted ?', 'N O !', 'Y E S !');
         if (cor== 1)
            P= load('P_synthetic.txt');
            N= load('N_synthetic.txt');
         else
            P= load('P_synthCorr.txt');
            N= load('N_synthCorr.txt');
         end
         % The corrupted data was produced by the
         % commented code given next with sigma= 0.1
%          if (cor== 2)
%              sigma= input('Noise variance..');
%              if isempty(sigma)
%                  sigma= 0.2;
%              end
%              P= P+ abs(sigma)* randn(size(P));
%              N= N+ abs(sigma)* randn(size(N));
%          end
         
     case 17
         database= 'wineQuality';
         disp([database d])
         ka= menu('Recognize with a quality at least of:',...
                  '1','2','3','4','5','6','7','8','9','10');
         disp(['Quality: No less than ' num2str(ka)])
         
         N= load('wine.txt');
         komp= (N(:,12) < ka);         
         P= N(~komp,1:11); N= N(komp,1:11);
     case 18
         database= 'user';
         fprintf('User provides ascii file P_user.txt for P\n')
         fprintf('and N_user.txt for N\n');
         P= load('P_user.txt');
         N= load('N_user.txt');
         
 end
 
 k= menu('Linear or quadratic separating surface ?',...
         '(Hyper)Planes', '(Hyper)Spheres',...
         '(Hyper)Quadratics with no cross terms');
 if (k== 1)
     disp('(Hyper) Planes');
 elseif (k== 2)
     disp('(Hyper) Spheres');
     P= [P sum(P.*P,2)]; N= [N sum(N.*N,2)];
 elseif (k== 3)
     disp('(Hyper) Quadratic with no cross terms')
     P= [P P.*P]; N= [N N.*N];
 end
 
 %Initial DataSet sizes
 [Prows n]= size(P); [Nrows n]= size(N);
 
 k= menu('Scale the data to [0 1] x ... [0 1] ?',...
         'NO. Do not scale the data', 'Please.. Scale the data');
 if (k== 2)
     disp('Data scaled');
     minmin= min(min([P; N]));
     P= P- minmin; N= N- minmin;
     MaxMax= max(max([P; N]));
     P= P/MaxMax; N= N/MaxMax;
 else
     disp('Data Unscaled');
     minmin= 0; MaxMax= 1; % No shift, no factor
 end

 %Prepare the testing set. 
 disp('Testing set:');
 disp('Ratio:  0 <= Ratio < 1')
 perc= input('Ratio: Sizeof(Testing) / Sizeof(DataSet) = ');
 if (isempty(perc)) | (perc< 0) | (perc> 1) 
     perc= 0.2;
 end
 VProws= floor(perc* Prows);  
 VP= P(1:VProws,:);
 
 VNrows= floor(perc* Nrows);  
 VN= N(1:VNrows,:);
 
 % The training set
 P= P(VProws+1:end,:); N= N(VNrows+1:end,:);
 
 %Number of individuals in testing and training set
 [Prows n]= size(P); [Nrows n]= size(N); 
 IniPop = Prows +  Nrows;  % In training
 IniVPop= VProws+ VNrows;  % In testing
 
 fprintf('Training %d: %d in P, %d in N\n',...
                  IniPop, Prows, Nrows);
 
 fprintf('Testing %d: %d in P, %d in N\n',...
                  IniVPop, VProws, VNrows);
 
 k= menu('What test?',...
         ['Processing time needed to classify ' int2str(IniPop)],...
         'Cross Validation',...
         'training & testing errors (%) at all iterations',...
         'The above + view graph!',...
         'Test error and planes at a given accuracy. Random test set');
     
 switch k
     
     case 1  % Processing time
         fprintf('Computing time.\n');
         tic;
         [W th c Pf Nf]= clpAug10(solver,P,N);
         fprintf('%1.3f seconds needed to classify %d\n',...
                 toc, Prows+ Nrows);
         failures= clValidate(VP,VN,W,th,c);
         fprintf('%1.3f%% error, %d planes.\n',...
                 failures, length(c));
         
     case 2  % Cross validation
         if (~perc | ~IniVPop)
             trials= 1;
         else
             trials= floor(1/perc);
         end
         fprintf('%d-fold validation\n',trials);
         
         %Initialization
         tfinal=      zeros(1,trials);
         NumPlane=    zeros(1,trials);
         failPN=      zeros(1,trials);
         failVPN=     zeros(1,trials);
         OnePlanePN=  zeros(1,trials);
         OnePlaneVPN= zeros(1,trials);
         
         for kros= 1: trials
             tic;
             [W th c]= clpAug10(solver,P,N);
             tfinal(kros)= toc;
             NumPlane(kros)= length(th);
             failVPN(kros)= clValidate(VP,VN,W,th,c);
             failPN(kros)=    clValidate(P,N,W,th,c);
             
             %Include OnePlane required by reviewer
             OnePlanePN(kros)=...
                 clValidate(P,N,W(:,1),th(1),c(1));
             OnePlaneVPN(kros)=...
                 clValidate(VP,VN,W(:,1),th(1),c(1));
            
             
             % New P, VP
             P= [P; VP]; 
             VP= P(1:VProws,:); 
             P= P(VProws+1:end,:);
             
             %New N, VN
             N= [N; VN];
             VN= N(1:VNrows,:);
             N= N(VNrows+1:end,:);
         end
         %Some statistics
         fprintf(...
         'timing: min= %1.3f, avg= %1.3f, max= %1.3f\n',...
          min(tfinal), sum(tfinal)/trials, max(tfinal));
      
         fprintf('Number of planes to termination:\n');
         fprintf('min= %d, avg= %1.3f, max= %d\n',...
          min(NumPlane), sum(NumPlane)/trials, max(NumPlane));
         
         fprintf('Errors after termination:\n');
         minfailPN= min(failPN);
         avgfailPN= sum(failPN) / trials;
         maxfailPN= max(failPN);
         fprintf(...
         'Train: min= %1.3f%%, avg= %1.3f%%, max= %1.3f%%\n',...
          minfailPN, avgfailPN, maxfailPN);
         
         minfailVPN= min(failVPN);
         avgfailVPN= sum(failVPN) / trials;
         maxfailVPN= max(failVPN);
         fprintf(...
         'Test: min= %1.3f%%, avg= %1.3f%%, max= %1.3f%%\n',...
          minfailVPN, avgfailVPN, maxfailVPN);
         
         %Again to satisfy the reviewer
         fprintf('Errors with only one plane:\n');
         minOnePlanePN= min(OnePlanePN);
         avgOnePlanePN= sum(OnePlanePN) / trials;
         maxOnePlanePN= max(OnePlanePN);
         fprintf(...
         'Train: min= %1.3f%%, avg= %1.3f%%, max= %1.3f%%\n',...
          minOnePlanePN, avgOnePlanePN, maxOnePlanePN);

         %Testing set    
         minOnePlaneVPN= min(OnePlaneVPN);
         avgOnePlaneVPN= sum(OnePlaneVPN) / trials;
         maxOnePlaneVPN= max(OnePlaneVPN);
         fprintf(...
         'Test: min= %1.3f%%, avg= %1.3f%%, max= %1.3f%%\n',...
         minOnePlaneVPN, avgOnePlaneVPN, maxOnePlaneVPN);
         
         
         
         
      case {3,4}  % Errors at each iteration
         [W th c]= clpAug10(solver,P,N);
         t= length(th);   %Total Number of Hyperplanes
         
         failTrain= []; failTest= [];  %Initialization
         for j= 1:t
            failTrain(j)=...
                clValidate(P,N,W(:,1:j),th(1:j),c(1:j));
            failTest(j)=...
              clValidate(VP,VN,W(:,1:j),th(1:j),c(1:j));
         end
         
  
         % The graph
         if (k== 5)
             hf= figure;
             set(hf,'color',[1 1 1]);
             hp= plot(1:t,failTrain, '-',...
                      1:t,failTest, ':');
             set(hp,'color', [0 0 0], 'LineWidth', 1.5)
             hx= xlabel('hyperplanes','FontSize',12,'FontWeight','bold');
             hy= ylabel(  'errors(%)','FontSize',12,'FontWeight','bold');
             ht= title(database,'FontSize',12,'FontWeight','bold');
             hl= legend('training','predicting',1);
             set(hl,'FontSize',12)
         end
         
     case 5  % Random population... Different acc
         trials= 5;
         for acc= input('Use brackets: [...], or colon x:v:z\naccuracy = ')
             fprintf('accuracy= %1.3f\n',acc);
             failTest= zeros(trials,1); planes= zeros(trials,1);             
             for kt= 1:trials
                 P= [P; VP]; N= [N; VN];   %Gather all individuals
             
                 % Declare the size of the train and testing
                 [Prows n]= size(P); [Nrows n]= size(N);  %Inefficient but sure
                 VProws= floor(0.2* Prows); VNrows= floor(0.2* Nrows);
             
                 %random individuals
                 P= P(randperm(Prows),:);
                 N= N(randperm(Nrows),:);
              
                 % Define testing and training
                 VP= P(1:VProws,:);      VN= N(1:VNrows,:);
                 P=  P(VProws+1:end,:);   N= N(VNrows+1:end,:);
                 
                 %Solve with the given accuracy
                 [W th c]= clpAug10(solver,P,N,acc);
                 
                 %Find the required results
                 failTest(kt)= clValidate(VP,VN,W,th,c);
                 planes(kt)= length(th);                 
             end
             
             %Determine range of values
             fprintf('errors: %1.3f - %1.3f\n',min(failTest),max(failTest));
             fprintf('planes: %1.3f - %1.3f\n',min(planes),max(planes));
         end
         
 end
 
 return
         
         
         
         
         
         
         
         