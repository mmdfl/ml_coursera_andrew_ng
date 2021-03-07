function J = computeCost(X, y, theta)
  m = length(y);
  J = 0;
  h=X*theta;
  J = h-y;
  J = J.*J;
  J = sum(J(:))/(2*m);

end
