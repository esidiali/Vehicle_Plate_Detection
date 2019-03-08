Dans ce l�t de la segmentation et de la detection des plaques d'immatriculation, nous vous proposons trois m�thodes:

1- m�thode non supervis�e : bas�e g�n�ralement sur les id�es du papier, avec des changements importants dans l'approche
   -----> les arguments d'entr�e : seulement le chemin de l'image � traiter ;
   -----> sortie : liste des coordonn�es de la plaque de l'image (X, Y, Delta_X, Delta_Y).

2- m�thode supervis�e � base des for�ts al�atoires de r�gression.
   -----> les arguments d'entr�e : array/tableau des caract�ristiques des images (apr�s HOG features extraction) avec les annotations correspondantes ;
   -----> sortie : array of coordinantes, same shape as the ground truth data: (X, Y, Delta_X, Delta_Y).

3- m�thode supervis�e bas�e sur les r�seaux de neurons convolutionnels pour l'extraction des caract�ristiques.
   -----> les arguments d'entr�e : comme RF ;
   -----> sortie :  comme RF.

les performances de ces m�thodes sont tr�s diff�rentes.Les m�thodes supervis�es sont de loins les meilleures, compte-tenu du fait que, dans la m�thode non supervis�e,
on doit d�finir nous m�me un ensemble de crit�res d'extraction de la plaque ou ces coordonn�es avec des hypoth�ses fixes mais des images variables.

DATA : pour le jeu des donn�es utilis�, vous pouvez le trouvez ici : https://github.com/openalpr/benchmarks/tree/master/endtoend/