
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tw4_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.191708471617300e+03, 2.191717286890702e+03, 2.191765720370614e+03, 2.191634376861519e+03, 2.191701883813813e+03, 2.191701883813813e+03, 5.923076480727825e+01, 5.923020472124387e+01, 5.922016012326413e+01, 5.926012560914855e+01, 5.923295856531377e+01, 5.923295856531377e+01, 2.298736709358296e+00, 2.296416646806762e+00, 2.246369189075721e+00, 2.280135370829837e+00, 2.280123493539976e+00, 2.280123493539976e+00, 1.854218336509532e-01, 1.891445807201684e-01, 3.096122914940911e+00, 1.195908281606454e-01, 1.778980274130396e-01, 1.778980274130395e-01, 3.075262329819505e-04, 3.410986855365098e-04, 1.019328664074255e-02, 1.021862260632350e-04, 2.046148941388157e-04, 2.046148941388157e-04, 1.283307077375580e+02, 1.283365058470793e+02, 1.283313478496035e+02, 1.283364604102343e+02, 1.283334421118759e+02, 1.283334421118759e+02, 2.037160190034483e+01, 2.061190648639007e+01, 2.028347932839054e+01, 2.049539600529818e+01, 2.054721554066003e+01, 2.054721554066003e+01, 1.670015736467878e+00, 1.878799069365698e+00, 1.444032582923619e+00, 1.485258051957923e+00, 1.718431212464394e+00, 1.718431212464394e+00, 6.365947520092455e-02, 1.997987088320782e-01, 5.520821007220724e-02, 1.702252497531877e+01, 8.336079040426736e-02, 8.336079040426736e-02, 6.092788440951140e-05, 9.764962313760038e-05, 5.730610466301245e-05, 2.592680121447311e-02, 9.191991051308627e-05, 9.191991051308630e-05, 1.586791532203853e+00, 1.587643096571332e+00, 1.587406934663512e+00, 1.587135466878874e+00, 1.587273531281511e+00, 1.587273531281511e+00, 1.496472371770700e+00, 1.235215948182243e+00, 1.307664263013401e+00, 1.379947583460033e+00, 1.342322427811223e+00, 1.342322427811223e+00, 2.068850044223153e+00, 3.001909829059097e-01, 4.045911532830493e-01, 6.254045531609489e-01, 5.008661031247287e-01, 5.008661031247287e-01, 1.049689856763285e+00, 9.324899593242293e-03, 1.713391840582947e-02, 5.631493263830523e-01, 4.072169274926362e-02, 4.072169274926363e-02, 6.103848452108024e-04, 6.976655942325803e-06, 3.088189945382037e-05, 3.616079394873704e-02, 7.733715371235291e-05, 7.733715371235278e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tw4_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.346526270014198e+03, 3.346517849677759e+03, 3.346550380137028e+03, 3.346535363390657e+03, 3.346628455823648e+03, 3.346646976459483e+03, 3.346356152475698e+03, 3.346299910926090e+03, 3.346540602217489e+03, 3.346439333429519e+03, 3.346540602217489e+03, 3.346439333429519e+03, 8.806316170020996e+01, 8.806335962787949e+01, 8.806403793130799e+01, 8.806406081422850e+01, 8.808354282176217e+01, 8.808968937968793e+01, 8.807818754667929e+01, 8.808351517890320e+01, 8.803268867107296e+01, 8.810420952733077e+01, 8.803268867107296e+01, 8.810420952733077e+01, 3.228295498635533e+00, 3.256477473927073e+00, 3.217489820212738e+00, 3.251901116469723e+00, 3.079839421333802e+00, 3.038042865851253e+00, 3.103059616950073e+00, 3.116434440522184e+00, 3.311223586618012e+00, 2.915600427521458e+00, 3.311223586618012e+00, 2.915600427521458e+00, 2.256721425512487e-01, 2.396474135674361e-01, 2.290365893733526e-01, 2.456290879460921e-01, 4.261749370971589e+00, 4.570847578346446e+00, 1.548287562157993e-01, 1.584519144195868e-01, 2.449486692211708e-01, 9.538703295779331e-02, 2.449486692211708e-01, 9.538703295779330e-02, 4.783685253884840e-04, 5.400479681482564e-04, 5.251686680780587e-04, 6.025184453067443e-04, 1.561370562321013e-02, 1.732801392972342e-02, 1.730395714712333e-04, 1.673251528977843e-04, 3.808254865936365e-04, 1.233988045092399e-04, 3.808254865936366e-04, 1.233988045092400e-04, 2.001791673977421e+02, 2.000820400009619e+02, 2.002844744960535e+02, 2.001837405085128e+02, 2.001852801825369e+02, 2.000857226978676e+02, 2.002756359506645e+02, 2.001781098900760e+02, 2.002328943682837e+02, 2.001330782021890e+02, 2.002328943682837e+02, 2.001330782021890e+02, 2.754973692068700e+01, 2.754696431665794e+01, 2.795329486177593e+01, 2.793815786192313e+01, 2.722557282603826e+01, 2.731442605480837e+01, 2.756946654691806e+01, 2.766435536604122e+01, 2.802665402555605e+01, 2.778583365268299e+01, 2.802665402555605e+01, 2.778583365268299e+01, 2.521992309624268e+00, 2.508817588251257e+00, 3.033026784099672e+00, 3.036715574286015e+00, 2.066829482647999e+00, 2.226187018741164e+00, 2.292695260440157e+00, 2.446654912161919e+00, 2.736299458537792e+00, 2.467874814511278e+00, 2.736299458537793e+00, 2.467874814511279e+00, 9.152093620605208e-02, 9.269685584522183e-02, 2.544058950436187e-01, 2.564267266490911e-01, 7.716829452214426e-02, 8.409849481393439e-02, 2.755590205950122e+01, 2.753158965740592e+01, 1.104668537053355e-01, 1.162434135606692e-01, 1.104668537053355e-01, 1.162434135606692e-01, 9.740170342427591e-05, 1.051960437845276e-04, 1.601955282680535e-04, 1.650822575332100e-04, 8.955901600156421e-05, 1.004317876242764e-04, 3.991637220178592e-02, 4.043904949515296e-02, 9.344197512046672e-05, 1.761487196555075e-04, 9.344197512046672e-05, 1.761487196555075e-04, 2.605328129177218e+00, 2.625821175604850e+00, 2.557928250514491e+00, 2.578454386527517e+00, 2.574300735233508e+00, 2.594921963806081e+00, 2.588160463699464e+00, 2.608569312601243e+00, 2.581204656996019e+00, 2.601716282478582e+00, 2.581204656996019e+00, 2.601716282478582e+00, 2.472770420373373e+00, 2.489188946023557e+00, 1.740313229738215e+00, 1.755790677442504e+00, 1.924312894478982e+00, 1.942238154985089e+00, 2.129666391034816e+00, 2.145048745319298e+00, 2.023228658668533e+00, 2.038656223172935e+00, 2.023228658668533e+00, 2.038656223172935e+00, 3.325957593403665e+00, 3.338369578887693e+00, 3.756252635639328e-01, 3.791160162802059e-01, 5.095743351243794e-01, 5.181715885663702e-01, 8.648010363210020e-01, 8.744806983025473e-01, 6.604538618228218e-01, 6.598228717596123e-01, 6.604538618228215e-01, 6.598228717596123e-01, 1.470564923825466e+00, 1.491439197830055e+00, 1.508366635031039e-02, 1.528254397072131e-02, 2.648930302436361e-02, 2.814176490793108e-02, 7.998626100818125e-01, 8.248670714892827e-01, 5.776249530941291e-02, 6.209579156125453e-02, 5.776249530941291e-02, 6.209579156125455e-02, 9.775512411949424e-04, 1.049460450678466e-03, 1.160063758594604e-05, 1.165342727813166e-05, 4.802168722053785e-05, 5.432007403374708e-05, 5.331181505302879e-02, 5.485919716941034e-02, 8.382399439146321e-05, 1.480056003990187e-04, 8.382399439146307e-05, 1.480056003990184e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tw4_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.551131275447241e-06, 0.000000000000000e+00, 1.551137859225545e-06, 1.551123656859166e-06, 0.000000000000000e+00, 1.551132320206697e-06, 1.551086400428642e-06, 0.000000000000000e+00, 1.551082541446189e-06, 1.551172766688706e-06, 0.000000000000000e+00, 1.551192518742562e-06, 1.551127382142345e-06, 0.000000000000000e+00, 1.551139080187939e-06, 1.551127382142345e-06, 0.000000000000000e+00, 1.551139080187939e-06, 3.409886685160274e-04, 0.000000000000000e+00, 3.410856476565133e-04, 3.409958667475787e-04, 0.000000000000000e+00, 3.411001089895235e-04, 3.412437093250550e-04, 0.000000000000000e+00, 3.413027046068656e-04, 3.406489606062065e-04, 0.000000000000000e+00, 3.407259335483384e-04, 3.411270366566263e-04, 0.000000000000000e+00, 3.409238938260717e-04, 3.411270366566263e-04, 0.000000000000000e+00, 3.409238938260717e-04, 4.191054516467783e-02, 0.000000000000000e+00, 4.208684160021041e-02, 4.187290789039656e-02, 0.000000000000000e+00, 4.209175841231561e-02, 4.128727570516935e-02, 0.000000000000000e+00, 4.087564407574851e-02, 4.022183798646189e-02, 0.000000000000000e+00, 4.036692810093313e-02, 4.210744190147805e-02, 0.000000000000000e+00, 3.738611438737610e-02, 4.210744190147805e-02, 0.000000000000000e+00, 3.738611438737610e-02, 1.139388878290862e+00, 0.000000000000000e+00, 1.111432460403355e+00, 1.158046354850380e+00, 0.000000000000000e+00, 1.119658231322967e+00, 2.788481130350872e-02, 0.000000000000000e+00, 2.679846313676431e-02, 1.167471070218367e+00, 0.000000000000000e+00, 1.146143143105157e+00, 1.025506863912970e+00, 0.000000000000000e+00, 1.213040307657514e+00, 1.025506863912969e+00, 0.000000000000000e+00, 1.213040307657516e+00, 1.517110070725546e-01, 0.000000000000000e+00, 1.595941899593658e-01, 1.675027376944246e-01, 0.000000000000000e+00, 1.785186082624454e-01, 5.155120611091285e-01, 0.000000000000000e+00, 5.472270763110554e-01, 8.353530726201590e-02, 0.000000000000000e+00, 7.991354368526754e-02, 1.351154687226339e-01, 0.000000000000000e+00, 1.950933133410065e-01, 1.351154687226342e-01, 0.000000000000000e+00, 1.950933133410069e-01, 1.112340608870699e-04, 0.000000000000000e+00, 1.113102379530319e-04, 1.112650914797856e-04, 0.000000000000000e+00, 1.113402795476021e-04, 1.112350434710878e-04, 0.000000000000000e+00, 1.113107389353273e-04, 1.112616813463446e-04, 0.000000000000000e+00, 1.113380395720823e-04, 1.112505581988365e-04, 0.000000000000000e+00, 1.113254475809922e-04, 1.112505581988365e-04, 0.000000000000000e+00, 1.113254475809922e-04, 1.484025180988159e-03, 0.000000000000000e+00, 1.484212160857947e-03, 1.465421387393722e-03, 0.000000000000000e+00, 1.466293467830441e-03, 1.473803368975515e-03, 0.000000000000000e+00, 1.477369742397499e-03, 1.458357933096490e-03, 0.000000000000000e+00, 1.461243178955694e-03, 1.479694445731178e-03, 0.000000000000000e+00, 1.475833147505308e-03, 1.479694445731178e-03, 0.000000000000000e+00, 1.475833147505308e-03, 7.259436905148406e-02, 0.000000000000000e+00, 7.320812338096207e-02, 6.429607973971309e-02, 0.000000000000000e+00, 6.429736034674902e-02, 9.423778271068026e-02, 0.000000000000000e+00, 8.599465439606349e-02, 9.583027312534312e-02, 0.000000000000000e+00, 8.672268068654861e-02, 6.590412614886068e-02, 0.000000000000000e+00, 7.631158919450082e-02, 6.590412614886069e-02, 0.000000000000000e+00, 7.631158919450086e-02, 9.298616372782923e-01, 0.000000000000000e+00, 9.534767678360498e-01, 7.059162026826119e-01, 0.000000000000000e+00, 7.026524180979420e-01, 9.515596703539742e-01, 0.000000000000000e+00, 9.712419851149324e-01, 2.358362464455133e-03, 0.000000000000000e+00, 2.361712669078133e-03, 1.151293592534731e+00, 0.000000000000000e+00, 1.443241381697804e+00, 1.151293592534731e+00, 0.000000000000000e+00, 1.443241381697804e+00, 8.797359636203246e-02, 0.000000000000000e+00, 7.913230010435349e-02, 9.715798708189413e-02, 0.000000000000000e+00, 9.106139267480656e-02, 4.138325800325982e-01, 0.000000000000000e+00, 4.875380366812400e-01, 9.791673163380482e-01, 0.000000000000000e+00, 9.493339249539753e-01, 2.101852990247215e-01, 0.000000000000000e+00, 2.835270696412919e-01, 2.101852990247208e-01, 0.000000000000000e+00, 2.835270696412914e-01, 8.427825247633508e-02, 0.000000000000000e+00, 8.334864077342284e-02, 8.329539117331539e-02, 0.000000000000000e+00, 8.240009862836806e-02, 8.363859650437952e-02, 0.000000000000000e+00, 8.273360423846450e-02, 8.392888636002992e-02, 0.000000000000000e+00, 8.300943424849372e-02, 8.378404164682365e-02, 0.000000000000000e+00, 8.287157460911207e-02, 8.378404164682365e-02, 0.000000000000000e+00, 8.287157460911207e-02, 9.224956767439496e-02, 0.000000000000000e+00, 9.134700157739258e-02, 1.076015573057846e-01, 0.000000000000000e+00, 1.066300768962680e-01, 1.039011920422939e-01, 0.000000000000000e+00, 1.029048848701025e-01, 9.976502057313311e-02, 0.000000000000000e+00, 9.882075613889009e-02, 1.020657999395228e-01, 0.000000000000000e+00, 1.010635312192250e-01, 1.020657999395228e-01, 0.000000000000000e+00, 1.010635312192250e-01, 5.562194925199903e-02, 0.000000000000000e+00, 5.546924631330182e-02, 5.197506151585433e-01, 0.000000000000000e+00, 5.171303468608328e-01, 4.239801591399507e-01, 0.000000000000000e+00, 4.212291689529256e-01, 2.917733905144340e-01, 0.000000000000000e+00, 2.880632979051282e-01, 3.609837713558307e-01, 0.000000000000000e+00, 3.628625311558969e-01, 3.609837713558311e-01, 0.000000000000000e+00, 3.628625311558970e-01, 1.370760646804854e-01, 0.000000000000000e+00, 1.352996739207706e-01, 4.628511538281218e-01, 0.000000000000000e+00, 4.665465346386314e-01, 6.115775583876821e-01, 0.000000000000000e+00, 6.453404160978948e-01, 3.628424421619533e-01, 0.000000000000000e+00, 3.505694100865922e-01, 1.200166361917937e+00, 0.000000000000000e+00, 1.474970342249472e+00, 1.200166361917938e+00, 0.000000000000000e+00, 1.474970342249473e+00, 1.719723767998609e-01, 0.000000000000000e+00, 1.819233799810976e-01, 9.340110449757980e-02, 0.000000000000000e+00, 1.658238587645296e-01, 1.172949322707234e-01, 0.000000000000000e+00, 1.328073697889857e-01, 1.284049676151533e+00, 0.000000000000000e+00, 1.277031847108326e+00, 4.104630549021087e-01, 0.000000000000000e+00, 2.697829031401668e-01, 4.104630549021092e-01, 0.000000000000000e+00, 2.697829031401673e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05