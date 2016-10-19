function [Failure] = clValidate (P, N, W, th, cl)
% Esta función es la usada en la tesis de Espinal (A6_ValidarRed). Hay que revisar su
% calidad. Está siendo modificada para usar con clTest
%
%   Esta función calcula la precisión de la RNA
%   En este caso, P y N contienen los sets de validación de P y N
%   W, th son las neuronas entrenadas con los sets de Entrenamiento(Hiperplanos)
%   cl es la clase de la neurona correspondiente (1 para P y -1 para N)
%
% Modificaciones: Se da como salida % de errores UNICAMENTE

[P_Filas P_Cols] = size(P);
[N_Filas N_Cols] = size(N);
Impr_P = [];
Impr_N = [];
[W_Filas, W_Cols] = size(W);
Comparar = +1e-07;

%disp('*** Validación de los puntos de P. Deben dar todos clase=+1 ***');
for n=1:P_Filas
    for i=1:W_Cols
               % ok=((P    * w(:)  )- th) >= -1e-07; % Los clasificados de P son 1 en ok 
        PruebaValid=(P(n, : )* W( : ,i))-th(i);
        if PruebaValid  >= Comparar    % Original: if P(n,:)* W(:,i)-th(i)  > -1e-07
            Impr_P = [Impr_P; [ n, PruebaValid, cl(i),i]];
            break
        elseif i==W_Cols
            sprintf(' No encontré al punto P(%d) en neurona %d, debe ser de clase %d ',n, i, -cl(i))
            Impr_P = [Impr_P; [ n, PruebaValid, -cl(i),i]];
        end
    end
end

%disp('*** Validación de los puntos de N.  Deben dar todos clase=-1 ***');
for n=1:N_Filas
    for i=1:W_Cols
        PruebaValid=(N(n,:)* W(:,i))-th(i);
        if PruebaValid  >= Comparar    % Original: if N(n,:)* W(:,i)-th(i)  > -1e-07
            Impr_N = [Impr_N; [ n, PruebaValid, cl(i),i]];
            break
        elseif i==W_Cols
            sprintf(' No encontré al punto N(%d) en neurona %d, debe ser de clase %d ',n, i, -cl(i))
            Impr_N = [Impr_N; [ n, PruebaValid, -cl(i),i]];
        end
    end
end
%Impr_P
%Impr_N
Errores_P = sum(Impr_P(:,3) == -1);
Aciertos_P = sum(Impr_P(:,3) == 1);
Precision_P = Aciertos_P / (Aciertos_P + Errores_P)

Errores_N = sum(Impr_N(:,3) == 1);
Aciertos_N = sum(Impr_N(:,3) == -1);
Precision_N = Aciertos_N / (Aciertos_N + Errores_N)
Nro_Neuronas=size(cl);
Cant_HP = Nro_Neuronas(2);

%%CODIGO AGREGADO%% 12/10/2016
Total_Fails = Errores_P + Errores_N;
Total_Correct = Aciertos_P + Aciertos_N;
Failure = Total_Fails / (Total_Fails + Total_Correct); % de errores


%FIN CODIGO AGREGADO% 12/10/2016

% Cálculo del promedio de aciertos de P y N por Hiperplano i
%TOT_Prec = [];
%for i=1:W_Cols
%    Error_P = sum(Impr_P(:,3) == -1 & Impr_P(:,4) == i);
%    Acier_P = sum(Impr_P(:,3) == +1 & Impr_P(:,4) == i);
%    Error_N = sum(Impr_N(:,3) == +1 & Impr_N(:,4) == i);
%    Acier_N = sum(Impr_N(:,3) == -1 & Impr_N(:,4) == i);
%    Sum_E_PN = Error_P + Error_N;
%    Sum_A_PN = Acier_P + Acier_N;
%    Prom_Prec = Sum_A_PN / (Sum_A_PN + Sum_E_PN);
%    TOT_Prec = [TOT_Prec; [i, Prom_Prec, Sum_A_PN, Sum_E_PN, (Sum_A_PN + Sum_E_PN)]];
%end
%disp('    HP           %Prom   Aciertos   Errores   Total')
%TOT_Prec
end