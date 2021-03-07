function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  m = length(y); % number of training examples
  n = rows(theta); % é a qtde de parametros
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters
    %updates theta
    t_aux=theta;
    for j = 1:n %j 
      soma=0;
      %calcula somatório
      for i = 1:m % m é a quantidade de amostras
        
        soma = soma+(X(i,:)*theta-y(i))*X(i,j);
      end
      t_aux(j)=theta(j)-(alpha/m)*soma;
    end
    theta=t_aux;
    J_history(iter) = computeCost(X, y, theta);

  end

end
