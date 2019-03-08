Dans ce lôt de la segmentation et de la detection des plaques d'immatriculation, nous vous proposons trois méthodes:

1- méthode non supervisée : basée généralement sur les idées du papier, avec des changements importants dans l'approche
   -----> les arguments d'entrée : seulement le chemin de l'image à traiter ;
   -----> sortie : liste des coordonnées de la plaque de l'image (X, Y, Delta_X, Delta_Y).

2- méthode supervisée à base des forêts aléatoires de régression.
   -----> les arguments d'entrée : array/tableau des caractéristiques des images (après HOG features extraction) avec les annotations correspondantes ;
   -----> sortie : array of coordinantes, same shape as the ground truth data: (X, Y, Delta_X, Delta_Y).

3- méthode supervisée basée sur les réseaux de neurons convolutionnels pour l'extraction des caractèristiques.
   -----> les arguments d'entrée : comme RF ;
   -----> sortie :  comme RF.

les performances de ces méthodes sont très différentes.Les méthodes supervisées sont de loins les meilleures, compte-tenu du fait que, dans la méthode non supervisée,
on doit définir nous même un ensemble de critères d'extraction de la plaque ou ces coordonnées avec des hypothèses fixes mais des images variables.

DATA : pour le jeu des données utilisé, vous pouvez le trouvez ici : https://github.com/openalpr/benchmarks/tree/master/endtoend/