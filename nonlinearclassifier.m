function classifier = nonlinearclassifier(X,y,C,sigma)
      %
      %kernel of dot product
      %
      kernel=X*X';
      %
      %size of X
      %
      n=size(X,1);
      %
      %Lets solve the dual of this optimization problem
      %
      cvx_begin
      cvx_precision high
      variable al(n);
      minimize ((0.5.*quad_form(y.*al, kernel))-ones(n,1)'*al);
      subject to
           al>=0;
           y'*al==0;
           al<=C; % Additional condition for non-linear case
      cvx_end
      alpha_star=al;
        %
        %
        %By using the calculated optimal alpha_star value we can determine
        %the support vectors.
        %
        %Let's find out support vectors and remove all other points outside
        %of margin
        %
        % tolarence of alpha_star value
        tol=10^-5;
    alpha_star(alpha_star<tol)=0;
    X_support = X(alpha_star>0,:);
    y_support = y(alpha_star>0);
    alpha_star = alpha_star(alpha_star>0);
    kernel=X*X_support';
    b_star=mean(y-kernel*(alpha_star.*y_support));
    classifier = new_model(X_support, y_support, alpha_star, b_star, sigma) 
end
