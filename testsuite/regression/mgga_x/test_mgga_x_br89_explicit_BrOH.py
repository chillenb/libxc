
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_br89_explicit_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.882000573040233e+01, -1.882003574325706e+01, -1.882024663576522e+01, -1.881975365189411e+01, -1.882002089514399e+01, -1.882002089514399e+01, -3.617641612787716e+00, -3.617580795419854e+00, -3.615952163306873e+00, -3.618200136141442e+00, -3.617630774828804e+00, -3.617630774828804e+00, -7.732692921036142e-01, -7.735321239141663e-01, -7.818634450195828e-01, -7.825259645148410e-01, -7.733436571689332e-01, -7.733436571689332e-01, -2.422980771231433e-01, -2.425812328242260e-01, -9.235066827315190e-01, -2.182911992524619e-01, -2.423282043316983e-01, -2.423282043316983e-01, -7.693323817882544e-02, -7.769078118134164e-02, -1.352022353604930e-01, -7.383856620288434e-02, -7.700048456903871e-02, -7.700048456903871e-02, -9.163379184553740e+00, -6.626267319169120e+00, -3.815495183773538e+00, -9.966620568384702e+00, -4.290404523308841e+00, -6.009088011409994e+00, -1.429808076246648e+00, -2.635801061090687e+00, -2.339175079076436e+00, -2.356500183026466e+00, -2.345332381415929e+00, -2.084660269239142e+00, -1.239440790033223e+00, -9.819602205530295e-01, -1.061861326283783e+00, -6.044181489561151e-01, -6.608053447467155e-01, -7.040049751837519e-01, -7.122855033040291e-02, -4.219738626747215e-01, -7.006947214669743e-02, -2.051205009023173e+00, -1.856062960252331e-01, -1.640381170822522e-01, -1.482668127673116e-01, -1.896454305546261e+00, -2.948498117977998e-03, -4.423414607922907e-02, -1.013676227923396e-01, -3.584422252449552e-03, -3.066652653406938e+01, -4.350030633293844e-01, -4.358380682684074e-01, -4.612791783977034e+00, -4.361631667400837e-01, -8.185473820644461e+01, -4.233001018367984e-01, -5.566911609874291e-01, -9.626405464194953e-01, -4.450251950804688e-01, -6.282722457157858e-01, -1.212606121870029e+00, -1.389811433339656e+00, -1.688587448224766e-01, -2.004829463298635e-01, -2.578316776768244e-01, -4.512289296354342e-01, -4.090468116151271e-01, -5.128986938429161e-01, -2.411076645696249e-01, -1.645391706411110e-01, -3.648888779089442e-01, -5.762238827915262e-02, -1.520286072462620e-01, -8.717530907332962e-03, -2.136046760796959e-01, -2.045987426497038e-03, -5.702805221723504e-02, -3.046381092217690e-03, -3.046284124305886e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_br89_explicit_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.824567435895229e+01, -2.824574144201391e+01, -2.824611882021860e+01, -2.824501872607813e+01, -2.824570905958561e+01, -2.824570905958561e+01, -4.577646185382790e+00, -4.577701843280014e+00, -4.579437524811443e+00, -4.577581916081066e+00, -4.577693695666385e+00, -4.577693695666385e+00, -8.675502685842318e-01, -8.665575644618937e-01, -8.419218584256248e-01, -8.463009424066040e-01, -8.671801792310156e-01, -8.671801792310156e-01, -2.369909734156422e-01, -2.395475464734242e-01, -1.056630783676040e+00, -1.867715090544488e-01, -2.377321732564143e-01, -2.377321732564143e-01, -4.079634355470802e-02, -4.180785604010503e-02, -8.518848914296147e-02, -3.064830874072542e-02, -4.144105576509870e-02, -4.144105576509871e-02, -8.666113260633143e+00, -7.290045688308751e+00, -5.111490448113987e+00, -9.120737477707818e+00, -6.386604458027020e+00, -7.058724723877351e+00, -1.909914563201089e+00, -2.718944113478429e+00, -2.521383887304376e+00, -2.541267609224796e+00, -2.554737401088578e+00, -2.429061977393093e+00, -1.087781688979604e+00, -9.800799485314198e-01, -9.669230412212519e-01, -7.797083570219828e-01, -8.033142831751684e-01, -8.142696558703673e-01, -9.539694779867638e-02, -3.503520042819584e-01, -9.349209290977251e-02, -2.612029871548687e+00, -1.498078113031550e-01, -1.257832009576083e-01, -9.221752024246742e-02, -1.291544225533085e+00, -3.931330854660825e-03, -5.910921420577560e-02, -5.536219570726050e-02, -4.779238016915072e-03, -2.101144588445387e+01, -5.800075403043213e-01, -5.811176066583246e-01, -3.277642796472589e+00, -5.815553454511799e-01, -5.652922875402653e+01, -5.644003756719018e-01, -6.518386709756623e-01, -8.933750735502242e-01, -6.579482093860739e-01, -7.240712254082777e-01, -1.043956366716822e+00, -1.229916340903224e+00, -2.254038960859349e-01, -2.675119991672796e-01, -3.465353452405957e-01, -4.486966769173317e-01, -4.220318065583664e-01, -5.892440540791192e-01, -1.730840626265427e-01, -1.117939747299115e-01, -4.326089218718102e-01, -7.690222512383681e-02, -1.102168889451652e-01, -1.162377313824826e-02, -1.463322667151958e-01, -2.727985812767818e-03, -7.485843432228916e-02, -4.061966900558016e-03, -4.061718536099255e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_explicit_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.654619510992060e-09, -4.654538124733332e-09, -4.654241916994808e-09, -4.655573451029473e-09, -4.654575986797804e-09, -4.654575986797804e-09, -9.784789952708422e-06, -9.785282617897908e-06, -9.798062734060795e-06, -9.778852704721775e-06, -9.784834347732600e-06, -9.784834347732600e-06, -4.780608598782210e-03, -4.775060933593484e-03, -4.621658656446106e-03, -4.594104894889588e-03, -4.779090141169113e-03, -4.779090141169113e-03, -5.333753075934213e-01, -5.255983611996993e-01, -2.348803558529576e-03, -1.048108178611045e+00, -5.313724204812168e-01, -5.313724204812181e-01, -1.688075569423213e+03, -1.476640427962692e+03, -2.477588374421837e+01, -1.338997325573261e+04, -1.536421440715561e+03, -1.536421440715561e+03, -3.738901120971471e-07, -9.225918127752952e-07, -5.173129422138360e-10, -3.099169437940180e-07, -5.649087423110263e-07, -1.310110656981955e-06, -3.876086756346842e-09, -3.833927909838409e-05, -5.757262940012138e-05, -5.593651493270179e-05, -5.680744323192790e-05, -8.968925727465595e-05, -1.601936058984063e-03, -2.356033553046296e-03, -2.437085119609152e-03, -1.266113509712162e-02, -8.945074008422064e-03, -6.966230744333071e-03, -4.864865772855258e-03, -1.383969689645870e-01, -1.015108283938159e-04, -9.606397656952553e-05, -2.007782058359824e+00, -2.792080482124193e+00, -5.346208650776670e+03, -6.766847999291785e+01, -4.107594764728987e-09, -8.057255213174234e-03, -8.615261022016852e+03, -8.874288802584051e-05, -9.351904400043744e-06, -4.672435167502258e-12, -1.286264364191845e-14, -1.381708224434842e-04, -7.646724517533046e-12, -1.949143931422442e-06, -2.652623178487948e-14, -1.776179729970460e-02, -3.289762134669494e-03, -3.826465918798827e-03, -1.099249723982641e-02, -1.970462279983206e-03, -9.821342650164343e-04, -7.802550698841660e-06, -1.670959010386697e-06, -7.364284414096803e-05, -4.854139845024921e-02, -6.524614899363901e-02, -2.465227869965708e-02, -9.884556610023745e+00, -1.062146118028473e+01, -9.622296008532956e-02, -4.028594869206504e-04, -5.800463531014056e+00, -1.039325971475417e-03, -4.804401787836223e+04, -2.478292284130654e-04, -1.177194232746093e+00, -9.367133081504626e-02, -1.372632468275096e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_explicit_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-2.249545214996782e-04, -2.249522262485893e-04, -2.249464244883506e-04, -2.249839254210826e-04, -2.249532718005806e-04, -2.249532718005806e-04, -2.040645789848292e-03, -2.040761385346618e-03, -2.044032773357398e-03, -2.040280047697137e-03, -2.040687735805969e-03, -2.040687735805969e-03, -7.022322590581272e-03, -6.992595566619802e-03, -6.241499350395454e-03, -6.351983646505598e-03, -7.012210264344375e-03, -7.012210264344375e-03, -1.604086891947355e-02, -1.644684091437568e-02, -5.939571059168454e-03, -1.201196341338221e-02, -1.617287951059705e-02, -1.617287951059706e-02, -8.844458089898713e-03, -8.891157826443105e-03, -1.001741739470401e-02, -7.745632586851319e-03, -8.964897452880583e-03, -8.964897452880583e-03, -2.613982474259821e-04, -6.453569429510804e-04, -3.616899848682428e-07, -2.167747073561673e-04, -3.950524478992664e-04, -9.161876658914974e-04, -1.436188506789103e-07, -1.450806622727921e-03, -2.115036155832389e-03, -2.089115388747515e-03, -2.158715096585955e-03, -3.408242699639118e-03, -1.552816728253890e-03, -2.927155075973672e-03, -2.018401242859501e-03, -1.189860842247096e-02, -9.709136332138496e-03, -7.561266005645493e-03, -2.211183689790587e-05, -4.252288948944485e-03, -4.438733059748892e-07, -3.358583273248162e-03, -1.410996336260286e-02, -1.962172793813779e-02, -2.776402073170329e-03, -5.245360461524610e-05, -1.340603311442905e-15, -8.825045958182195e-06, -5.051661193876529e-03, -5.203545225729815e-11, -9.959904101204332e-06, -4.896932730431980e-12, -1.355862070630175e-14, -1.462953333964785e-04, -8.078384829797098e-12, -2.059173797941230e-06, -2.561712308527210e-14, -1.119807885614297e-02, -2.363755341429571e-03, -3.088655347742298e-03, -8.366881528081415e-03, -1.499807013136272e-03, -1.406498160047827e-03, -4.769917563770690e-07, -1.711277878280314e-07, -1.575897516944431e-05, -7.186909694389911e-03, -9.660170404950201e-03, -1.140362064282398e-02, -3.175157620196810e-03, -8.464947632153491e-03, -1.736802120993516e-02, -9.791011257595821e-07, -1.409732315490867e-02, -8.766075146003562e-09, -7.642691368623608e-04, -2.702530001665831e-11, -2.484255658459486e-03, -3.371524964228723e-08, -4.940534733358751e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_explicit_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.198544687989694e-04, 7.198471239954857e-04, 7.198285583627205e-04, 7.199485613474636e-04, 7.198504697618584e-04, 7.198504697618584e-04, 6.530066527514260e-03, 6.530436433108674e-03, 6.540904874742771e-03, 6.528896152630513e-03, 6.530200754578995e-03, 6.530200754578995e-03, 2.247143228986014e-02, 2.237630581318317e-02, 1.997279792126550e-02, 2.032634766881783e-02, 2.243907284590193e-02, 2.243907284590193e-02, 5.133078054231530e-02, 5.262989092600209e-02, 1.900662738933894e-02, 3.843828292282308e-02, 5.175321443391045e-02, 5.175321443391059e-02, 2.830226588767589e-02, 2.845170504461796e-02, 3.205573566305264e-02, 2.478602427792425e-02, 2.868767184921788e-02, 2.868767184921788e-02, 8.364743917631451e-04, 2.065142217443465e-03, 1.157407951578318e-06, 6.936790635397358e-04, 1.264167833277656e-03, 2.931800530852782e-03, 4.595803221725030e-07, 4.642581192729359e-03, 6.768115698663634e-03, 6.685169243992044e-03, 6.907888309075069e-03, 1.090637663884509e-02, 4.969013530412475e-03, 9.366896243115784e-03, 6.458883977150403e-03, 3.807554695190722e-02, 3.106923626284335e-02, 2.419605121806566e-02, 7.075787807328910e-05, 1.360732463662235e-02, 1.420394579119568e-06, 1.074746647439415e-02, 4.515188276032915e-02, 6.278952940204101e-02, 8.884486634145051e-03, 1.678515347687876e-04, 4.289930666187636e-15, 2.824014706618209e-05, 1.616531582040488e-02, 1.665134472233541e-10, 3.187169312385386e-05, 1.567018473596197e-11, 4.338758629588019e-14, 4.681450668687304e-04, 2.585083145563707e-11, 6.589356153411938e-06, 8.197479388923078e-14, 3.583385233965735e-02, 7.564017092574620e-03, 9.883697112775341e-03, 2.677402088986057e-02, 4.799382442036070e-03, 4.500794112153053e-03, 1.526373620406417e-06, 5.476089210498994e-07, 5.042872054222084e-05, 2.299811102204773e-02, 3.091254529584069e-02, 3.649158605703677e-02, 1.016050438462979e-02, 2.708783242289119e-02, 5.557766787179251e-02, 3.133123602430318e-06, 4.511143409570777e-02, 2.805144046747988e-08, 2.445661237959554e-03, 8.648096011281378e-11, 7.949618107070270e-03, 1.078887988546906e-07, 1.580971114445651e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05