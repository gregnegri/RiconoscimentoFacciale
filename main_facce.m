close all
clear all
clc

%% 1. TRAINING SET

load datiORL.mat

% Estraggo il training set dal database
% N.B.: il database è costituito dalle 10 immagini per ciascuno di 40 individui
% ogni immagine è 112x92 pixel disposta sulle colonne della matrice V

N = 10; % numero di soggetti da includere nel training set
M = 8; % numero di immagini per ogni soggetto

for i = 1:N
    
    X(:,(i-1)*M + 1: i*M) = V(:,(i-1)*10 + 1: (i-1)*10 + M);
    
end

% Visualizzo graficamente il training set

facce = [];
for j = 1:N
    
    temp = [];
    for i = 1:M
        
        col = X(:, (j-1)*M + i);
        w1 = reshape(col,112,92);
        temp = [temp w1];
        
    end
    facce = [facce;temp];
    
end

figure(1);
imshow(facce,[]);
title('Faces');
clear facce temp col

%% 2. SPAZIO DELLE FACCE

% Calcolo la decomposizione SVD di A

[m,n] = size(X);
Xbar = mean(X')';%sum(X,2)/n;
A = X - repmat(Xbar,1,n);
[U,Sigma,~] = svd(A);

figure(2);
plot(diag(Sigma).^2,'b','LineWidth',4); % quadrati dei valori singolari dal più grande al più piccolo (l'ultimo è 0)
title('Valori singolari'); 
axis square;
axis tight;

% Visualizzo graficamente le prime 25 autofacce

autofacce = [];
for j = 1:5
    
    temp = [];
    for i = 1:5
        
        col = U(:,5*(i-1) + j);
        w1 = reshape(col,112,92);
        temp = [temp w1];
        
    end
    autofacce = [autofacce; temp];
    
end

figure(3);
imshow(autofacce,[]);
title('Autofacce');

clear autofacce temp col

% Memorizzo le prime 15 autofacce

q = 15;
Uq = U(:,1:q);

%% 3. CLASSIFICAZIONE DI NUOVE IMMAGINI

% Calcolo la distanza massima delle immagini del training set dalle loro
% proiezioni nello spazio delle facce e definisco una soglia per la
% classificazione di nuove immagini in facce e non-facce

Xp = repmat(Xbar,1,n) + Uq*(Uq'*A); % proiezione del training set
dist_max = max(sqrt(sum((Xp - X).^2)));
dist_tol = 2*dist_max;

% Memorizzo le immagini da classificare

F1 = V(:,20); % immagine non presente in X di un soggetto noto
F2 = V(:,N*10 + 1); % immagine di un soggetto non noto
load ./bottiglia.mat 
F3 = K(:);
load ./gatto.mat
F4 = K(:);
load ./fiore.mat
F5 = K(:);

test_image = {F1,F2,F3,F4,F5};
num_test = length(test_image);

clc;

% Classifico le immagini

for i = 1 : num_test
    
    z = test_image{i};
    
    % Calcolo la proiezione della nuova immagine e la distanza di quest'ultima
    % dalla sua proiezione nello spazio delle facce
    zp = Xbar + Uq*(Uq'*(z-Xbar));
    dist_z = norm(zp-z);

    % Visualizzo graficamente l'immagine da classificare e la sua proiezione 
    figure(i+3);
    subplot(1,2,1);
    imshow(reshape(z,112,92),[]);
    subplot(1,2,2);
    imshow(reshape(zp,112,92),[]);
    title(['Errore = ' num2str(dist_z)]);
    
    % Classificazione: z è una faccia?
    fprintf('L''immagine %d è una ',i);
    if dist_z <= dist_tol
        
        fprintf('faccia: l''errore è %f <= %f\n\n',dist_z,dist_tol);
        
    else
        
        fprintf('non-faccia: l''errore è %f > %f\n\n',dist_z,dist_tol);
        
    end
    
end