#tests the decision C++ library from Python

import unittest
import xml.dom, xml.dom.minidom

class NbcSerializableTestCase(unittest.TestCase):
    """test case for create xml and use with faif-nbc classifier"""

    def testNBCXMLDescription(self):
        '''generate xml understood by faif-nbc-serializable'''
        def addSimpleTextTag(xmlDoc, parent, name, value):
            tag = xmlDoc.createElement(name)
            val = xmlDoc.createTextNode(value)
            tag.appendChild(val)
            parent.appendChild(tag)
            return tag


        NBC_CLASS_ID = 0
        NBCBASE_CLASS_ID = 2
        NBCIMPL_CLASS_ID = 1

        class_id = 3
        object_id = 0

        xmlDoc = xml.dom.minidom.Document()
        xmlRoot = xmlDoc.createElement('boost_serialization')
        xmlRoot.setAttribute('signature', 'serialization::archive')
        xmlRoot.setAttribute('version', '7')
        dt = xml.dom.minidom.getDOMImplementation('').createDocumentType('boost_serialization', '', '')
        xmlDoc.insertBefore(dt, xmlDoc.documentElement)
        xmlDoc.appendChild(xmlRoot)

        nbcEl = xmlDoc.createElement('NBC')
        xmlRoot.appendChild(nbcEl)
        nbcEl.setAttribute('class_id', str(NBC_CLASS_ID))
        nbcEl.setAttribute('tracking_level', "1")
        nbcEl.setAttribute('version', "0")
        nbcEl.setAttribute('object_id', '_' + str(object_id))
        NBC_object_id = object_id
        object_id += 1

        nbcBaseEl = xmlDoc.createElement('NBCBase')
        nbcBaseEl.setAttribute('class_id', str(NBCBASE_CLASS_ID))
        nbcBaseEl.setAttribute('tracking_level', "0")
        nbcBaseEl.setAttribute('version', "0")
        nbcEl.appendChild(nbcBaseEl)

        addSimpleTextTag(xmlDoc, nbcBaseEl, 'ClassifierDomainsCount', '0')

        clCat = xmlDoc.createElement('ClassifierCategory')
        clCat.setAttribute('object_id', '_' + str(object_id))
        object_id_ref = object_id
        object_id += 1
        nbcBaseEl.appendChild(clCat)

        addSimpleTextTag(xmlDoc, clCat, 'DomainId', '')

        domVals = xmlDoc.createElement('DomainValues')
        clCat.appendChild(domVals)
        addSimpleTextTag(xmlDoc, domVals, 'count', '0')
        addSimpleTextTag(xmlDoc, domVals, 'item_version', '0')


        nbcImplEl = xmlDoc.createElement('NBCImpl')
        nbcImplEl.setAttribute('class_id', str(NBCIMPL_CLASS_ID))
        nbcImplEl.setAttribute('tracking_level', "1")
        nbcImplEl.setAttribute('version', "0")
        nbcImplEl.setAttribute('object_id', '_' + str(object_id))
        object_id += 1
        nbcEl.appendChild(nbcImplEl)


        base = xmlDoc.createElement('Base')
        base.setAttribute('class_id', str(class_id))
        class_id += 1
        base.setAttribute('tracking_level', "1")
        base.setAttribute('version', "0")
        base.setAttribute('object_id', '_' + str(object_id))
        object_id += 1
        nbcImplEl.appendChild(base)

        parent = xmlDoc.createElement('Parent')
        parent.setAttribute('class_id_reference', str(NBC_CLASS_ID))
        parent.setAttribute('object_id_reference', '_' + str(NBC_object_id))
        parent.appendChild(xmlDoc.createTextNode(''))
        base.appendChild(parent)

        intProb = xmlDoc.createElement('InternalProb')
        intProb.setAttribute('class_id', str(class_id))
        class_id += class_id + 1
        intProb.setAttribute('tracking_level', "0")
        intProb.setAttribute('version', "0")
        nbcImplEl.appendChild(intProb)

        addSimpleTextTag(xmlDoc, intProb, 'count', '0')
        addSimpleTextTag(xmlDoc, intProb, 'item_version', '0')

        docTxt = xmlDoc.toprettyxml(encoding='UTF-8')
        self.assertTrue( len(docTxt) > 0 )

        f = open('x.xml', 'w')
        f.write(docTxt)
        f.close()

if __name__ == "__main__":
    #runs test if executed as a script
    print 'start C++ library python interface tests'
    unittest.main()


